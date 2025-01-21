from typing import TypeVar, Generic, List, Dict, Tuple, Optional, Union, Any
from collections.abc import Hashable
import json
import ast
import numpy as np

T = TypeVar("T")


class Archive(Generic[T]):
    """
    A flexible archive class for storing and manipulating collections of data.

    The class supports various operations like adding, updating,
    retrieving, and analyzing collections of data.
    """

    def __init__(self, name: str):
        """
        Initialize an Archive instance.

        Args:
            name (str): Name of the archive
        """
        self.name = name
        self._archive: Dict[Tuple[Hashable, ...], List[T]] = {}

    def add(self, key: Tuple[Hashable, ...], value: List[T]) -> None:
        """
        Add a new key-value pair to the archive.

        Args:
            key (Tuple[Hashable, ...]): Unique key for the entry
            value (List[T]): List of values to store

        Raises:
            TypeError: If key is not a tuple
        """
        if not isinstance(key, tuple):
            raise TypeError("Key must be a tuple")
        if not isinstance(value, list):
            value = list(value)
        self._archive[key] = value

    def update(self, key: Tuple[Hashable, ...], value: List[T]) -> None:
        """
        Update the value for an existing key.

        Args:
            key (Tuple[Hashable, ...]): Key to update
            value (List[T]): New values to set

        Raises:
            KeyError: If the key does not exist
        """
        if key not in self._archive:
            raise KeyError(f"Key {key} does not exist in the archive")

        if not isinstance(value, list):
            value = list(value)
        self._archive[key] = value

    def delete(self, key: Tuple[Hashable, ...]) -> None:
        """
        Delete a key-value pair from the archive.

        Args:
            key (Tuple[Hashable, ...]): Key to delete
        """
        self._archive.pop(key, None)

    def flatten_values(self) -> List[T]:
        """
        Return a flattened list of all values in the archive.

        Returns:
            List[T]: Flattened list of all values
        """
        return [item for sublist in self._archive.values() for item in sublist]

    def get(self, key: Tuple[Hashable, ...]) -> Optional[List[T]]:
        """
        Retrieve the value for a given key.

        Args:
            key (Tuple[Hashable, ...]): Key to retrieve

        Returns:
            Optional[List[T]]: Values associated with the key, or None
        """
        return self._archive.get(key)

    def keys(self) -> List[Tuple[Hashable, ...]]:
        """
        Return a list of all keys in the archive.

        Returns:
            List[Tuple[Hashable, ...]]: List of keys
        """
        return list(self._archive.keys())

    def exists(self, key: Tuple[Hashable, ...]) -> bool:
        """
        Check if a key exists in the archive.

        Args:
            key (Tuple[Hashable, ...]): Key to check

        Returns:
            bool: Whether the key exists
        """
        return key in self._archive

    def extend(self, key: Tuple[Hashable, ...], new_values: List[T]) -> None:
        """
        Extend the value list for an existing key.

        Args:
            key (Tuple[Hashable, ...]): Key to extend
            new_values (List[T]): Values to add

        Raises:
            KeyError: If the key does not exist
        """
        if key not in self._archive:
            raise KeyError(f"Key {key} does not exist in the archive")
        self._archive[key].extend(new_values)

    def values_are_numeric(self) -> bool:
        """
        Check if all values in the archive are numeric.

        Returns:
            bool: Whether all values are numeric
        """
        return all(isinstance(v, (int, float)) for v in self.flatten_values())

    def len_elements(self) -> Dict[Tuple[Hashable, ...], int]:
        """
        Return a dictionary of key lengths.

        Returns:
            Dict[Tuple[Hashable, ...], int]: Length of each key's value list
        """
        return {k: len(v) for k, v in self._archive.items()}

    def median_elements(self) -> Dict[Tuple[Hashable, ...], float]:
        """
        Calculate the median for each key's numeric values.

        Returns:
            Dict[Tuple[Hashable, ...], float]: Median of each key's values

        Raises:
            ValueError: If values are not numeric
        """
        if not self.values_are_numeric():
            raise ValueError("Values in the archive must be numeric")

        return {k: float(np.median(v)) for k, v in self._archive.items()}

    def idx_median_elements(self) -> Dict[Tuple[Hashable, ...], int]:
        """
        Find the index of the median for each key's numeric values.

        Returns:
            Dict[Tuple[Hashable, ...], int]: Index of median for each key

        Raises:
            ValueError: If values are not numeric
        """
        if not self.values_are_numeric():
            raise ValueError("Values in the archive must be numeric")

        return {k: int(np.argsort(v)[len(v) // 2]) for k, v in self._archive.items()}

    def idx_max_elements(self, seed: int = 0) -> Dict[Tuple[Hashable, ...], int]:
        """
        Find the index of max values for each key, with random tie-breaking.

        Args:
            seed (int, optional): Random seed for tie-breaking. Defaults to 0.

        Returns:
            Dict[Tuple[Hashable, ...], int]: Index of max for each key

        Raises:
            ValueError: If values are not numeric
        """
        if not self.values_are_numeric():
            raise ValueError("Values in the archive must be numeric")

        np.random.seed(seed)
        return {
            k: int(np.random.choice(np.where(v == np.max(v))[0]))
            for k, v in self._archive.items()
        }

    def subtract(self, archive_instance: "Archive[T]") -> "Archive[T]":
        """
        Subtract another archive, keeping elements from n to end.

        Args:
            archive_instance (Archive[T]): Archive to subtract

        Returns:
            Archive[T]: Resulting archive after subtraction
        """
        result = Archive[T](self.name)
        for key, value in self._archive.items():
            if archive_instance.exists(key):
                result.add(key, value[len(archive_instance.get(key)) :])
            else:
                result.add(key, value)

        # Remove empty keys
        result._archive = {k: v for k, v in result._archive.items() if v}

        return result

    def save(self, filepath: str) -> None:
        """
        Save the archive to a JSON file.

        Args:
            filepath (str): Path to save the JSON file
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._archive.items()}, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load the archive from a JSON file.

        Args:
            filepath (str): Path to the JSON file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self._archive = {ast.literal_eval(k): v for k, v in json.load(f).items()}

    def load_from_dict(self, data: Dict[str, List[T]]) -> None:
        """
        Load the archive from a dictionary.

        Args:
            data (Dict[str, List[T]]): Dictionary to load from
        """
        self._archive = {ast.literal_eval(k): v for k, v in data.items()}
