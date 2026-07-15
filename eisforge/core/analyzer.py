from __future__ import annotations
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from eisforge.core.fitter import CNLSFitter, FitResult
from eisforge.core.validators import KKValidationResult, KramersKronigValidator
from eisforge.parsers.base_parser import BaseEISParser, EISDataset
from eisforge.parsers.gamry_parser import GamryParser
from eisforge.parsers.generic_csv_parser import GenericCSVParser

logger = logging.getLogger(__name__)

_PARSER_REGISTRY = {
    ".dta": GamryParser,
    ".csv": GenericCSVParser,
    ".txt": GenericCSVParser,
}

try:
    from eisforge.parsers.autolab_parser import AutolabIDFParser
    _PARSER_REGISTRY[".idf"] = AutolabIDFParser
except ImportError:
    pass


class EisAnalyzer:
    def __init__(self, log_level=logging.INFO, kk_threshold=0.005, default_weight="modulus"):
        logging.basicConfig(level=log_level, format="[EISForge] %(levelname)s: %(message)s")
        self._logger = logging.getLogger(self.__class__.__name__)
        self.kk_threshold  = kk_threshold
        self.default_weight = default_weight
        self.datasets:    dict[str, EISDataset]          = {}
        self.fit_results: dict[str, FitResult]           = {}
        self.kk_results:  dict[str, KKValidationResult]  = {}

    def load(self, filepath, label=None, parser=None) -> EISDataset:
        path   = Path(filepath).resolve()
        label  = label or path.stem
        suffix = path.suffix.lower()
        if parser is None:
            parser = self._select_parser(suffix)
        dataset = parser.parse(path)
        dataset.validate_shapes()
        self.datasets[label] = dataset
        self._logger.info("Loaded: %s (%d points)", label, len(dataset.frequency))
        return dataset

    def load_from_arrays(self, frequency, z_real, z_imag, label="manual", metadata=None):
        dataset = EISDataset(
            frequency=np.asarray(frequency, dtype=np.float64),
            z_real=np.asarray(z_real, dtype=np.float64),
            z_imag=np.asarray(z_imag, dtype=np.float64),
            metadata=metadata or {},
        )
        dataset.validate_shapes()
        self.datasets[label] = dataset
        return dataset

    def validate_kk(self, dataset, label=None, mu=0.85) -> KKValidationResult:
        validator = KramersKronigValidator(residual_threshold=self.kk_threshold, mu=mu)
        result = validator.validate(dataset)
        if label:
            self.kk_results[label] = result
        if not result.passed:
            warnings.warn(result.warning_message or "K-K failed.", stacklevel=2)
        return result

    def fit(self, dataset, circuit, initial_guess, bounds=None, label=None, weight_by_modulus=None):
        use_mod = weight_by_modulus if weight_by_modulus is not None else (self.default_weight == "modulus")
        fitter  = CNLSFitter(circuit, initial_guess, bounds=bounds, weight_by_modulus=use_mod)
        result  = fitter.fit(dataset)
        if label:
            self.fit_results[label] = result
        self._logger.info("Fit: χ²=%.6f converged=%s", result.chi_squared, result.converged)
        return result

    def export_fit(self, fit_result, output_path):
        out = Path(output_path).resolve()
        rows = [{"parameter": n, "value": v,
                 "error": fit_result.parameter_errors.get(n, float("nan"))}
                for n, v in fit_result.parameters.items()]
        pd.DataFrame(rows).to_csv(out, index=False)
        return out

    def list_datasets(self):
        return list(self.datasets.keys())

    @staticmethod
    def _select_parser(suffix) -> BaseEISParser:
        cls = _PARSER_REGISTRY.get(suffix)
        if cls is None:
            raise ValueError(f"No parser for '{suffix}'. Supported: {list(_PARSER_REGISTRY)}")
        return cls()
