import os
import json
import matplotlib.pyplot as plt


bleu_scores_dir = "/mt-exercise-5/beam_experiments/bleu_scores"

beam_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
times_taken = [293.1242, 92.0636, 143.0650, 181.5066, 237.9304, 289.7589, 331.5274, 379.4675, 427.9370, 499.6236]


bleu_scores = []


for beam_size in beam_sizes:
    json_file = os.path.join(bleu_scores_dir, f"bleu.{beam_size}.json")
    with open(json_file, 'r') as f:
        bleu_data = json.load(f)
        bleu_scores.append(bleu_data["score"])

# Plot BLEU score vs Beam size
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(beam_sizes, bleu_scores, marker='o', linestyle='-', color='b')
plt.title('BLEU Score vs Beam Size')
plt.xlabel('Beam Size')
plt.ylabel('BLEU Score')
plt.grid(True)

# Plot Time taken vs Beam size
plt.subplot(1, 2, 2)
plt.plot(beam_sizes, times_taken, marker='o', linestyle='-', color='r')
plt.title('Time Taken vs Beam Size')
plt.xlabel('Beam Size')
plt.ylabel('Time Taken (seconds)')
plt.grid(True)

plt.tight_layout()
plt.savefig("beamsearch_analysis.png")
plt.show()

