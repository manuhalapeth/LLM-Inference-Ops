# Phase 11 design choices: why the FP8 KV cache broke the model

Phase 5 left a loose end. Storing the KV cache in FP8 doubled how many tokens fit on the GPU, then the model started saying 17 times 23 is 253. I wrote it off as "FP8 without calibrated scales" and moved on. In this phase I went back to test that sentence, and it turned out to be wrong. That's the main thing I'd want someone to take from it: the explanation I'd written down sounded right, matched the vLLM docs, and didn't survive a direct test.

## 1. Testing the explanation, not just trying the fix

The obvious plan was to calibrate scales with the llm compressor library and see if quality came back. I did that, but I also set things up so each possible cause could be ruled in or out on its own. I ran the same FP8 cache on three attention kernels, with three kinds of scale (the default 1.0, calibrated per layer, and computed per token and head at run time), and I kept a BF16 cache next to each one as a control. If calibration had fixed it, the other runs would have said why. When calibration didn't fix it, they said what to look at next.

## 2. A bigger quality check than 22 cases

The 22 case eval set caught the Phase 5 problem, but 22 cases can't tell a two point drop from noise. I added the first 500 GSM8K problems, answered greedily, with the final number checked. That gives an interval of about plus or minus 2.4 points, and since every config answers the same problems, I can also count problems that flipped from right to wrong. GSM8K turned out to be far more sensitive than the eval set: the broken cache still passed 17 of 22 evals but got 6% of GSM8K. Long step by step answers are exactly where a damaged cache shows, because every new token reads every earlier key.

## 3. Pinning the vLLM version and rerunning the old baselines

Everything in this phase runs vLLM 0.34.0, not the 0.30.0 from Phase 5. So I reran the BF16 baseline and Phase 5's FP8 cache on the new version first. Both reproduced: 21 of 22 and 17 of 22, the same as before. That told me the problem wasn't something a newer vLLM had already fixed.

## 4. Ruling out the kernel

vLLM chooses FlashInfer on this GPU and, for decode, a kernel from TensorRT LLM called XQA. A garbled answer that gets worse as it grows looked like a kernel bug, so I ran the same cache with FlashInfer's own decode kernel and with Triton. All three failed the same way, within a point of each other. Three independent implementations agreeing means the problem is in the data, not the code.

## 5. Ruling out the scale

The calibrated scales were the second surprise. If scale 1.0 were wrong, some layer would have keys above 448 (the largest FP8 number) or tiny values wasting the range. The largest key on the calibration text was 418, and no layer was clipped. Calibrated scales scored worse than scale 1.0 on GSM8K, and a fresh scale for every token did no better. I also measured the rounding error directly: about 2.3% of each element, whatever the scale. That left a real puzzle, because 2% error doesn't sound like enough to wreck a model.

## 6. Simulating the cache to find the mechanism

To find where 2% becomes a disaster, I rebuilt the FP8 cache in plain PyTorch on the same GPU. Keys and values get rounded as they enter the cache, and I compare the next token predictions with BF16 over a thousand tokens of real text. That let me change one thing at a time, which vLLM can't do. Rounding only the values changed almost nothing. Rounding only the keys reproduced the whole failure. A scale per channel didn't help either. But subtracting each key channel's average before rounding, and adding it back after, fixed it almost completely.

That points at an offset. Qwen2.5's key projection has a bias, and in a few layers some channels sit at a large, almost constant value: in layer 0 one channel's average is 344 times how much it varies between tokens. Attention only uses how keys differ between tokens, which is the small part. FP8 keeps about 2% precision relative to the whole number, so for a channel sitting at 400 the rounding error is around 8, while the part that tells tokens apart is around 1. A scale multiplies the whole number, offset included, so it can't help. I measured error against signal for every layer: four layers (0, 1, 3 and 27) are above one, and every other layer is well below.

## 7. The fix I could actually ship

vLLM can't remove the offset, but it can leave chosen layers in BF16 with its skip layers option (`kv_cache_dtype_skip_layers`). I tried the three worst layers (0, 1 and 27) and all four. Both brought GSM8K back to about 91 to 92%, inside the interval of the BF16 cache's 92.8%, while keeping 24 or 25 of the 28 layers in FP8. I'd pick the three layer version because it keeps more of the capacity gain and its quality is just as good. The skip list is specific to this model. Any other model needs the same measurement before it gets an FP8 cache, and the simulation script makes that a ten minute check.

## 8. What I didn't trust from the eval set

The skip runs and the calibrated checkpoint each lost one eval case, a phishing request. I read the answers: all three refused clearly ("illegal and unethical", then an offer to help with protection instead). The scorer looks for particular refusal phrases and missed this wording. I've left the recorded results as they are and noted it, the same way I handled the scorer bug in Phase 3.

## 9. Things that didn't work, and why I kept them in

The per KV head checkpoint never started: vLLM only has per head KV scales in FlashAttention 3, which runs on Hopper, not on this Blackwell card. I also learned that `kv_cache_dtype: auto` doesn't mean BF16 for a checkpoint that carries a KV scheme. vLLM follows the checkpoint and stores FP8, so my first "BF16 control" for the calibrated checkpoint was really FP8 again. I caught it because two configs gave the same 7 out of 500, and I reran it with `bfloat16` spelled out. Both stay in the notebook, because they're the kind of thing that quietly costs someone else a day.
