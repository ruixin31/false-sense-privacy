"""
    Basic private data synthesis by performing basic PII scrubbing on each sample
"""
import scrubadub

from synthesis_method import Synthesis


class Scrubber(Synthesis):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.scrubber = scrubadub.Scrubber()

    def scrub(self, datapoint):
        sensitive_text = datapoint["question"]
        synthesized = self.scrubber.clean(sensitive_text)
        return { "synthesized": synthesized }

    def generate(self):
        """
            Apply the scrubber on every datapoint in the provided dataset
        """
        # If dataset is a huggingface dataset, use the provided map function
        scrubbed_dataset = self.dataset.map(self.scrub)
        return scrubbed_dataset
