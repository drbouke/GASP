"""A minimal end-to-end example: detect an unsupported sentence in a RAG answer."""
from gasp import GASP

# The retrieved context and the answer a RAG system produced.
context = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 as the entrance arch to the World's Fair. "
    "The tower is 330 metres tall and was the tallest structure in the world until 1930."
)
answer = (
    "The Eiffel Tower was completed in 1889 in Paris. "
    "It is 330 metres tall. "
    "Visitors can dine for free at a rooftop restaurant that seats two thousand people."
)
# The third sentence is not supported by the context (a baseless addition).

detector = GASP("Qwen/Qwen2.5-1.5B-Instruct", k_chunks=3, threshold=0.5)
result = detector.detect(context=context, answer=answer, query="Describe the Eiffel Tower.")

print(f"{'verdict':8s} {'sensitivity':>11s}  sentence")
print("-" * 78)
for s in result:
    verdict = "FLAG" if s.flagged else "ok"
    print(f"{verdict:8s} {s.sensitivity:>11.2f}  {s.text.strip()}")
    if s.supporting_chunk:
        print(f"{'':8s} {'support ->':>11s}  {s.supporting_chunk.strip()[:64]}...")

print("\nsummary:", result.summary())
