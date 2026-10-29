"""Calculation Agent.

Verifies numeric safety factors against code requirements (e.g. ACI 318
flexural/shear/axial checks) using values extracted from retrieved clauses
and default material properties. Returns a structured calculation report.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings

logger = logging.getLogger("structuramind.agents.calculation")


class CalculationAgent(BaseAgent):
    """Performs structural code-compliance calculations."""

    def __init__(self) -> None:
        super().__init__("CalculationAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(
        self,
        query: str,
        context: str,
        concrete_strength_mpa: Optional[float] = None,
        steel_yield_mpa: Optional[float] = None,
    ) -> dict:
        """Run a code-compliance calculation and return a structured report."""
        fc = concrete_strength_mpa or settings.concrete_strength_default_mpa
        fy = steel_yield_mpa or settings.steel_yield_default_mpa

        # Extract numeric inputs from context/query.
        inputs = self._extract_numbers(query + " " + context)

        checks: List[dict] = []
        # Flexural strength check (phi * Mn >= Mu) - ACI 318-19 Table 21.2.1
        flexural = self._check_flexural(fc, fy, inputs)
        checks.append(flexural)

        # Shear strength check (phi * Vn >= Vu)
        shear = self._check_shear(fc, inputs)
        checks.append(shear)

        # Overall result.
        all_pass = all(c["pass"] for c in checks)
        calculation_summary = self._llm_calculation_summary(query, context, fc, fy, checks)

        return {
            "concrete_strength_mpa": fc,
            "steel_yield_mpa": fy,
            "inputs_extracted": inputs,
            "checks": checks,
            "all_pass": all_pass,
            "summary": calculation_summary,
            "units": settings.default_units,
        }

    # ------------------------------------------------------------------ #
    # Numeric extraction
    # ------------------------------------------------------------------ #
    def _extract_numbers(self, text: str) -> Dict[str, Optional[float]]:
        """Pull common engineering quantities out of free text."""
        numbers: Dict[str, Optional[float]] = {
            "factored_moment_knm": None,
            "factored_shear_kn": None,
            "factored_axial_kn": None,
            "width_mm": None,
            "effective_depth_mm": None,
        }

        patterns = {
            "factored_moment_knm": [
                r"M[u]?\s*[:=]\s*([\d\.]+)\s*kN\.?m",
                r"moment\s+([\d\.]+)\s*kN\.?m",
            ],
            "factored_shear_kn": [
                r"V[u]?\s*[:=]\s*([\d\.]+)\s*kN\b",
                r"shear\s+([\d\.]+)\s*kN\b",
            ],
            "factored_axial_kn": [
                r"P[u]?\s*[:=]\s*([\d\.]+)\s*kN\b",
                r"axial\s+([\d\.]+)\s*kN\b",
            ],
            "width_mm": [
                r"b\s*[:=]\s*([\d\.]+)\s*mm\b",
                r"width\s+([\d\.]+)\s*mm\b",
            ],
            "effective_depth_mm": [
                r"d\s*[:=]\s*([\d\.]+)\s*mm\b",
                r"effective depth\s+([\d\.]+)\s*mm\b",
            ],
        }

        for key, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        numbers[key] = float(m.group(1))
                    except ValueError:
                        pass
                    break
        return numbers

    # ------------------------------------------------------------------ #
    # Code checks (ACI 318 style, SI units)
    # ------------------------------------------------------------------ #
    def _check_flexural(self, fc: float, fy: float, inputs: Dict[str, Optional[float]]) -> dict:
        Mu = inputs["factored_moment_knm"]
        b = inputs["width_mm"]
        d = inputs["effective_depth_mm"]
        details: List[str] = []
        result = "not_applicable"
        pass_flag = False

        if Mu is not None and b is not None and d is not None:
            # phi = 0.9 (tension controlled, ACI 318-19 Table 21.2.1)
            phi = 0.9
            beta1 = 0.85 if fc <= 28 else (0.85 - 0.05 * ((fc - 28) / 7))
            beta1 = max(0.65, min(0.85, beta1))
            # Approximate lever arm ~ 0.9d; compute required steel area.
            approx_lever = 0.9 * d
            As_req = (Mu * 1e6) / (phi * fy * approx_lever)  # mm^2
            a = As_req * fy / (0.85 * fc * b)
            Mn = As_req * fy * (d - a / 2)
            phiMn = phi * Mn / 1e6  # kN.m
            capacity_ratio = phiMn / Mu if Mu else 0
            pass_flag = capacity_ratio >= 1.0
            result = "pass" if pass_flag else "fail"
            details = [
                f"fc' = {fc} MPa, fy = {fy} MPa, b = {b} mm, d = {d} mm",
                f"Required As ≈ {As_req:.1f} mm²",
                f"phi*Mn ≈ {phiMn:.2f} kN.m vs Mu = {Mu} kN.m",
                f"Capacity ratio = {capacity_ratio:.2f}",
                "ACI 318-19 Table 21.2.1 (phi = 0.9 tension-controlled)",
            ]
        elif Mu is None:
            details = ["No factored moment (Mu) provided; flexural check skipped."]
        else:
            details = ["Missing section dimensions (b, d); flexural check skipped."]

        return {
            "check": "flexural_strength",
            "code_reference": "ACI 318-19 Section 22.4 / Table 21.2.1",
            "result": result,
            "pass": pass_flag,
            "details": "; ".join(details),
        }


    def _check_shear(self, fc: float, inputs: Dict[str, Optional[float]]) -> dict:
        Vu = inputs["factored_shear_kn"]
        b = inputs["width_mm"]
        d = inputs["effective_depth_mm"]
        details: List[str] = []
        result = "not_applicable"
        pass_flag = False

        if Vu is not None and b is not None and d is not None:
            phi = 0.75  # shear (ACI 318-19 Table 21.2.1)
            # Concrete shear capacity Vc = 0.17 * lambda * sqrt(fc') * b * d (SI)
            lam = 1.0  # normal weight concrete
            Vc = 0.17 * lam * (fc ** 0.5) * b * d / 1000  # kN
            phiVc = phi * Vc
            pass_flag = phiVc >= Vu
            result = "pass" if pass_flag else "fail (shear reinforcement likely required)"
            details = [
                f"fc' = {fc} MPa, b = {b} mm, d = {d} mm",
                f"Vc ≈ {Vc:.2f} kN, phi*Vc = {phiVc:.2f} kN vs Vu = {Vu} kN",
                "ACI 318-19 Section 22.5 / Table 21.2.1 (phi = 0.75)",
            ]
        elif Vu is None:
            details = ["No factored shear (Vu) provided; shear check skipped."]
        else:
            details = ["Missing section dimensions (b, d); shear check skipped."]

        return {
            "check": "shear_strength",
            "code_reference": "ACI 318-19 Section 22.5 / Table 21.2.1",
            "result": result,
            "pass": pass_flag,
            "details": "; ".join(details),
        }

    # ------------------------------------------------------------------ #
    # LLM summary
    # ------------------------------------------------------------------ #
    def _llm_calculation_summary(
        self, query: str, context: str, fc: float, fy: float, checks: List[dict]
    ) -> str:
        system_prompt = (
            "You are a structural engineering calculation agent. Summarize the "
            "code-compliance checks below concisely in markdown. State pass/fail, "
            "the governing code clause, and any assumptions. Be precise."
        )
        checks_text = "\n".join(
            f"- {c['check']} ({c['code_reference']}): {c['result'].upper()} - {c['details']}"
            for c in checks
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Materials: fc' = {fc} MPa, fy = {fy} MPa\n"
            f"Context: {context[:1500]}\n\n"
            f"Checks:\n{checks_text}\n"
        )
        return self.call_llm(system_prompt, user_prompt, temperature=0.1)

