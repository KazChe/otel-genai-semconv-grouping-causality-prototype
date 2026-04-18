# OTel Semantic Conventions — Precedent Analysis for Propagation Guidance

There is precedent in the semantic-conventions repo, and it is directly relevant to our proposals.

The semantic-conventions repo does contain examples where they go beyond pure attribute naming and talk about **propagation behavior, parent/child structure, and instrumentation guidance**. But the pattern is important:

**the repo tends to allow this kind of guidance when it is tightly tied to trace semantics or interoperability, while often avoiding over-specifying the exact transport mechanism unless there is a mature standard.**

The strongest precedents I found are these.

First, the **messaging semconv** is very close in spirit to what I'm doing. It has an explicit **Context propagation** section that says a message creation context is created by the producer, should be propagated to consumers, and that producers should attach it to messages. It also discusses how send/process spans should be related, including when process spans may use the message creation context as parent and when links should be added instead. At the same time, it explicitly says it does **not** yet specify the exact attachment/extraction mechanism and defers that to future standards work. That is probably my best precedent. ([GitHub][1])

Second, the repo already has a precedent for **mechanism-level propagation guidance** in the **database semconv**. The database doc has a “Context propagation” section for **SQL commenter**, saying instrumentations may propagate context by injecting comments into SQL queries, should not enable it by default, should append the comment at the end of the query, and should allow users to override the propagator. That is definitely more than just naming attributes. ([GitHub][2])

Third, the **AWS Lambda semconv doc** goes even farther into runtime behavior than my draft does in some places. It defines how to apply semconv when instrumenting Lambda, says users must be able to configure a special `xray-lambda` propagator, says Lambda SDK instrumentations should provide that propagator, and even gives pseudocode for its behavior and configuration guidance for `OTEL_PROPAGATORS`. That is a pretty strong precedent for domain-specific instrumentation guidance living alongside semconv material. ([GitHub][3])

Fourth, there is also a **non-normative AWS compatibility** doc in the repo that talks about context propagation through AWS-managed services. It says an AWS-supported propagation format should be used, notes that implementations may allow configurable propagators, explains that some service-specific implementations are required, and warns about signing, attribute limits, and billing impacts. That is another precedent for “interoperability and operational consequences” guidance being documented near semconv work. ([GitHub][4])

So the answer is:

**Yes, there is precedent for semconv-adjacent material that talks about propagation and runtime behavior.**
But the safest precedent is not “specify every implementation detail.” It is more like:

* define the **semantic intent**
* define the **trace structure implications**
* give **interoperability guidance**
* avoid locking down the exact mechanism unless there is already a stable standard or a very strong reason

That pattern matches messaging especially well. Messaging says, in effect: “you need a creation context, it must propagate, here is what the trace structure should look like, but we are not yet standardizing the exact wire mechanism.” ([GitHub][1])

### Implications for our proposals

For **grouping**, the current wording standardizes the representation and gives interoperability guidance about baggage continuity across boundaries, similar in spirit to the messaging and AWS examples. ([GitHub][1])

For **causality**, the part that is most in-bounds is:

* standardizing the causal intent
* explaining why parent/child matters
* saying tool-call arguments are not a reliable general-purpose carrier
* recommending sidecars when available
* documenting out-of-band correlation as fallback

The part that would be most likely to feel out of bounds is if I tried to fully standardize a storage interface or lifecycle contract for out-of-band correlation. That would go beyond the repo’s usual precedent. Messaging is the clearest comparison here: it discusses attachment/extraction and parent-vs-link structure, but explicitly avoids standardizing the exact mechanism until the standards ecosystem is ready. ([GitHub][1])

### Summary

**There is precedent in semconv for specifying trace-structure intent and propagation/interoperability guidance. There is less precedent for standardizing framework-internal plumbing in detail.**

| Precedent | What it standardizes | What it defers |
|-----------|---------------------|----------------|
| Messaging | Creation context, parent/child structure | Attachment/extraction mechanism |
| SQL commenter | Injection approach, comment placement | |
| AWS Lambda | Propagator config, pseudocode | |
| **Our grouping proposal** | Attribute namespace, baggage transport | Boundary-specific plumbing |
| **Our causality proposal** | Causal intent, sidecar recommendation | Out-of-band storage contract |

How each precedent maps to our proposals:

* Messaging precedent supports the **causal tree / parent-child / propagation** reasoning.
* SQL commenter supports the **mechanism recommendation** style.
* AWS Lambda/AWS compatibility supports the **runtime-specific guidance** style.
* None of these strongly support standardizing a full-blown framework-internal out-of-band API contract, which is why ISSUE_CAUSALITY.md documents out-of-band correlation as a fallback pattern rather than specifying a contract.

### What our proposals deliberately avoid

Framework-internal API contracts, storage lifecycle management, and mandatory propagator configuration. This aligns with the observed pattern where semconv material specifies intent and structure but avoids locking implementation details.

[1]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/messaging/messaging-spans.md "Messaging spans (verified April 2026)"
[2]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/db/database-spans.md "Database spans, SQL commenter (verified April 2026)"
[3]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/faas/aws-lambda.md?plain=1 "AWS Lambda semconv (verified April 2026)"
[4]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/non-normative/compatibility/aws.md?plain=1 "AWS compatibility, non-normative (verified April 2026)"
