# Phase 8 design choices: the final report

Phase 8 didn't need a GPU. The job was to take eight phases of results and turn them into one story someone can read in a few minutes, without bending any of the numbers to make the story neater.

## 1. Recomputing everything instead of copying it

Every number in the final report is calculated from the JSON files in results, using the same functions the earlier phases used. I didn't type in a single result. That way the report can't drift from the data, and anyone can rerun it and get the same tables. It also meant the report could check the earlier write ups, which is how I found the pricing mistake below.

## 2. One SLO and one traffic mix for the whole journey

The roadmap asked for one chart from the first baseline to the final cluster. That only means something if every stage is judged the same way, so every bar uses Phase 4's traffic mix and the same interactive target: first token within half a second for 95% of requests, and at least 25 tokens a second while streaming. Capacity is the highest user step that met it. Where a sweep never failed the target, the table says so with a plus sign rather than pretending that was the limit.

## 3. A reference price instead of the price I happened to pay

The machines cost anywhere from 60 cents to $1.33 per GPU hour depending on the host and the day. If I charted the price I actually paid, the two GPU run from Phase 6 would look like the cheapest way to make tokens, mostly because that host was cheap. So the chart uses one reference price, a dollar per GPU hour, and the real price sits in its own column. The point of the chart is what the engineering changed, not which listing I found.

## 4. Fixing a mistake from Phase 5

While recomputing costs, the Phase 5 numbers didn't match the price written next to them. The tables said $1.211 an hour, which is what that machine cost, but the script had used $1.327, the price of the earlier machine, because that was its default. Every percentage was still right, since baseline and tuned configs used the same price. I changed the labels to the price that was really used, added a note that the true figures are 9% lower, and wrote the correction into the specification instead of quietly editing it.

## 5. Leaving Mooncake out of the chart

The roadmap's journey had a step called Mooncake at scale. I measured that step on a different workload, long conversations, because that's the only traffic where a shared cache could help, and on that workload it made things slower. Putting it on the same chart as the mix traffic would compare two different things. So the report gives Mooncake its own section with both results side by side: the case where it helped a lot, and the case where it hurt.

## 6. Being honest about what HAMi saves

It's tempting to say GPU sharing makes tokens cheaper. It doesn't. The 7B model's cost per token was the same shared or not; what changed is that the small model no longer needs a whole GPU to itself. The report says exactly that, and the chart shows it: the HAMi bar matches the one GPU each bar on cost, with half the hardware.

## 7. A capacity plan in streams, not people

Locust users send their next request the moment the last one finishes. That's a good way to size a GPU, because it measures how many requests can be in flight at once within the target, but it isn't how a person uses a chatbot. So the capacity plan talks about concurrent streams and says plainly that a real user, reading between requests, is only a fraction of one. It also adds a spare GPU, because Phase 6 showed a crash halves capacity until the server reloads and Phase 7 showed a new pod takes over two minutes to be ready.

## 8. A README that stands on its own

Most people will only read the README. So it opens with the two minute summary: the eight headline results in one table, the recommended deployment, the cost per million tokens, and the journey chart. The notebooks and the specification are linked for anyone who wants to check the work.
