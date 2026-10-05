# Phase 5 design choices: profiling and tuning one GPU

Phase 5 was about turning the Phase 4 breaking point into a decision: which engine settings this GPU should actually run with, and what it can promise. The biggest lesson wasn't about any one setting. It was that a speed result means nothing until the answers are checked.

## 1. Questioning the defaults first

Before tuning anything, I read how vLLM picks its defaults. It looks at the GPU's memory: 70 GB or more gets room for 1,024 requests and 8,192 tokens per scheduler step, and anything smaller falls through to a fallback of 256 requests and 2,048 tokens that the code itself marks as not yet tuned. A 32 GB RTX 5090 lands in that fallback. That told me the Phase 4 cap of 256 wasn't a law of the hardware, it was a guess, and it was worth testing.

## 2. One change per config, same load, same machine, same session

Every config is the baseline plus exactly one change, in its own file, so the file itself documents the experiment. All of them ran through the Phase 4 load, on one machine, one after the other, with the baseline rerun first in the same session. If two things change at once you can't tell which helped, and if the machine changes you can't tell whether the tuning or the hardware did it.

The machine was a different one from Phase 4, with the same GPU and the same server CPU. Its baseline came within 3% of Phase 4's, which is a nice check that the results reproduce.

## 3. Starting the sweep where the interesting part is

Phase 4 showed that nothing happens below 64 users, so Phase 5 swept 64 to 768 users at 45 seconds per step instead of 1 to 512 at 60. That cut each config to about seven minutes and added a step past the old limit, where the differences between configs show up.

## 4. Profiling to explain the numbers, not just measure them

vLLM can capture a PyTorch profiler trace on request. I captured 60 engine steps of the baseline at 128 users and wrote a small analyzer that groups every GPU kernel by the kind of work it does. The GPU was busy 98% of the time, and 73% of that was matrix multiplies on the model weights. That one fact explains most of Phase 5: settings that change scheduling barely move throughput, and settings that make the weights smaller move it a lot.

## 5. Capacity and cost at stated targets

"How many users" only means something next to a latency target, so I defined two: interactive (95% of first tokens within half a second, and at least 25 tokens a second while streaming) and relaxed (within a second, at least 16 tokens a second). For every config the comparison reports the highest load that meets each target and what a million output tokens costs at that load, using the machine's hourly price.

## 6. Running speculative decoding separately

Speculative decoding guesses several tokens ahead and has the model check them in one step. It helps when the GPU has spare capacity, which is only true at low load, so it got its own sweep from 1 to 64 users against the baseline instead of the high load sweep. It still made things worse: the guesses copied from the prompt rarely matched what the model wrote, and every wrong guess costs a verification step.

## 7. Checking quality before recommending anything

After the single changes, the obvious move was to combine the winners: FP8 weights, the FP8 KV cache and a higher request cap. It nearly doubled throughput. Then I ran the Phase 3 eval set on it and it got 17 times 23 wrong, broke JSON, and stopped refusing a phishing request. 15 of 22 passed, against 21 for the baseline.

Instead of throwing out both FP8 changes, I ran the evals on each one alone. FP8 weights passed exactly what the baseline passed. The FP8 KV cache was the problem: storing attention values in 8 bits without calibrated scaling factors loses too much precision for this model. That turned a confusing result into a clear recommendation, and it cost six minutes.

The rule I'm keeping from this phase: no throughput number goes into a recommendation without the eval set run on the same config.

## 8. Recommending FP8 weights, and saying what's left

The recommended config is FP8 weights alone: 50% more throughput, the cost per million tokens down from 6.4 to 4.4 cents, and the same eval results as BF16. I annotated each config file with its result, so anyone opening the folder sees which one to use and why the others aren't recommended.

What's left to try is written down rather than guessed at: FP8 weights with a higher request cap, which now has the memory to work; an FP8 KV cache with proper calibration; and finer load steps to pin down the interactive capacity between 128 and 192 users.

## 9. Running long sessions in the background

This was the longest GPU session yet, close to two hours of measurements. I started part of it as a background job on the machine, so a dropped SSH connection couldn't stop it, and copied results back after each config instead of waiting for the end. When the plan changed mid session, I copied the new config and scripts directly to the machine rather than pushing half finished code to GitHub.
