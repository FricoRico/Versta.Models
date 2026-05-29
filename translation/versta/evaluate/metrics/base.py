from abc import ABC, abstractmethod
from typing import List


class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> float:
        """
        Compute the metric score for a set of references and hypotheses.

        Args:
            references: List of reference translations.
            hypotheses: List of generated translations.

        Returns:
            Metric score as a float.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric.

        Returns:
            Metric name as a string.
        """
        pass
