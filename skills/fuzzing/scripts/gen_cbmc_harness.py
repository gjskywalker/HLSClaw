#!/usr/bin/env python3
"""
Generate a CBMC equivalence harness for comparing original and optimized
software-style C/C++ code.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

POINTER_ARRAY_EXTENT = 100


class FunctionSignatureParser:
    """Parse C/C++ function signatures."""

    @staticmethod
    def extract_function_signature(code: str, func_name: str) -> str:
        pattern = rf"([\w\s\*&]+?)\s+{func_name}\s*\(\s*(.*?)\s*\)\s*[{{;]"
        match = re.search(pattern, code, re.DOTALL)
        if not match:
            return None

        return_type = match.group(1).strip()
        params = match.group(2).strip()
        return f"{return_type} {func_name}({params})"

    @staticmethod
    def parse_parameters(signature: str) -> List[Tuple[str, str]]:
        match = re.search(r"\((.*)\)", signature, re.DOTALL)
        if not match:
            return []

        param_str = match.group(1).strip()
        if not param_str or param_str == "void":
            return []

        params: List[Tuple[str, str]] = []
        parts: List[str] = []
        current = ""
        depth = 0

        for char in param_str:
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            parts.append(current.strip())

        for part in parts:
            parsed = re.match(r"([\w\s\*&]+?)\s+(\w+)\s*(\[.*\])?$", part)
            if not parsed:
                continue
            param_type = parsed.group(1).strip()
            param_name = parsed.group(2).strip()
            array_part = parsed.group(3) if parsed.group(3) else ""
            if array_part:
                param_type += " " + param_name + array_part
            params.append((param_type, param_name))

        return params

    @staticmethod
    def extract_return_type(signature: str) -> str:
        match = re.match(r"([\w\s\*&]+?)\s+\w+\s*\(", signature)
        if match:
            return match.group(1).strip()
        return "void"


class CBMCHarnessGenerator:
    """Generate a CBMC harness for software-equivalence checking."""

    def __init__(self, optimized_code_path: str, original_code_path: str, function_name: str, output_dir: str = "."):
        self.optimized_code_path = optimized_code_path
        self.original_code_path = original_code_path
        self.function_name = function_name
        self.output_dir = output_dir
        self.parser = FunctionSignatureParser()
        self.define_values: Dict[str, int] = {}

    def _strip_structured_artifacts(self, code: str) -> str:
        optimized_blocks = re.findall(
            r"<optimized_code>(.*?)</optimized_code>",
            code,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if optimized_blocks:
            code = optimized_blocks[-1]
        elif re.search(r"<optimized_code>", code, flags=re.IGNORECASE):
            code = re.split(r"<optimized_code>", code, flags=re.IGNORECASE)[-1]

        block_tags = ("analysis", "think", "json", "command", "reference", "skill")
        for tag in block_tags:
            code = re.sub(
                rf"<{tag}>.*?</{tag}>",
                "",
                code,
                flags=re.DOTALL | re.IGNORECASE,
            )

        code = re.sub(
            r"</?(optimized_code|analysis|think|json|command|reference|skill)>",
            "",
            code,
            flags=re.IGNORECASE,
        )
        code = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    def preprocess_code(self, code: str) -> Tuple[str, List[str]]:
        defines: List[str] = []
        code = self._strip_structured_artifacts(code)

        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#define"):
                defines.append(" ".join(stripped.split()))

        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        code = re.sub(r"#pragma\s+HLS\s+.*?$", "", code, flags=re.MULTILINE)

        lines = []
        for line in code.splitlines():
            if not line.strip().startswith("#define"):
                lines.append(line)

        return "\n".join(lines), defines

    def read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return ""

    def _storage_decl(self, param_type: str, param_name: str, suffix: str = "") -> str:
        if "[" in param_type:
            target_name = f"{param_name}{suffix}"
            return re.sub(rf"\b{re.escape(param_name)}\b(?=\s*\[)", target_name, param_type, count=1)
        if "*" in param_type:
            base_type = param_type.replace("*", "").strip()
            return f"{base_type} {param_name}{suffix}[{POINTER_ARRAY_EXTENT}]"
        return f"{param_type} {param_name}{suffix}"

    def _element_type(self, param_type: str, param_name: str) -> str:
        if "[" in param_type:
            return re.sub(rf"\b{re.escape(param_name)}\b(\s*\[[^\]]*\])+", "", param_type).strip()
        if "*" in param_type:
            return param_type.replace("*", "").strip()
        return param_type.strip()

    def _extent(self, param_type: str) -> int:
        if "[" in param_type:
            dims = re.findall(r"\[([^\]]+)\]", param_type)
            if not dims:
                return 0
            extent = 1
            for dim in dims:
                token = dim.strip()
                if token.isdigit():
                    extent *= int(token)
                    continue
                if token in self.define_values:
                    extent *= self.define_values[token]
                    continue
                return 0
            return extent
        if "*" in param_type:
            return POINTER_ARRAY_EXTENT
        return 0

    def _load_define_values(self, defines: List[str]) -> None:
        values: Dict[str, int] = {}
        for define in defines:
            match = re.match(r"#define\s+([A-Za-z_]\w*)\s+(-?\d+)\b", define.strip())
            if not match:
                continue
            values[match.group(1)] = int(match.group(2))
        self.define_values = values

    def _generate_input_struct(self, params: List[Tuple[str, str]]) -> str:
        members = [f"    {self._storage_decl(param_type, param_name)};" for param_type, param_name in params]
        return "struct SymbolicInput {\n%s\n};" % "\n".join(members)

    def _generate_input_assumptions(self, params: List[Tuple[str, str]]) -> str:
        max_index = 0
        min_index = 0
        for param_type, _ in params:
            extent = self._extent(param_type)
            if extent <= 0:
                continue
            max_index = max(max_index, extent)
            if min_index == 0:
                min_index = extent
            else:
                min_index = min(min_index, extent)
        if max_index <= 0:
            max_index = POINTER_ARRAY_EXTENT
        if min_index <= 0:
            min_index = max_index

        lines: List[str] = []
        for param_type, param_name in params:
            lowered = param_name.lower()
            if not any(token in lowered for token in ("addr", "idx", "index", "col", "cols")):
                continue
            index_bound = min_index if any(token in lowered for token in ("col", "cols")) else max_index
            extent = self._extent(param_type)
            if extent > 0:
                lines.append(f"    for (int i = 0; i < {extent}; ++i) {{")
                lines.append(f"        __CPROVER_assume(input->{param_name}[i] >= 0);")
                lines.append(f"        __CPROVER_assume(input->{param_name}[i] < {index_bound});")
                lines.append("    }")
            else:
                lines.append(f"    __CPROVER_assume(input->{param_name} >= 0);")
                lines.append(f"    __CPROVER_assume(input->{param_name} < {index_bound});")

        if not lines:
            return "void assume_legal_input(struct SymbolicInput *input) {\n}\n"
        return "void assume_legal_input(struct SymbolicInput *input) {\n%s\n}\n" % "\n".join(lines)

    def _defined_functions(self, code: str) -> List[str]:
        names: List[str] = []
        seen = set()
        control_keywords = {"if", "for", "while", "switch", "catch"}
        pattern = re.compile(
            r"^\s*(?!if\b|for\b|while\b|switch\b)(?:[A-Za-z_][\w\s\*&]*?\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            name = match.group(1)
            if name in control_keywords:
                continue
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _rename_defined_functions(self, code: str, suffix: str) -> str:
        renamed = code
        # Rename every function defined in the translation unit so helper
        # functions do not collide when original/optimized code are concatenated.
        for name in sorted(self._defined_functions(code), key=len, reverse=True):
            renamed = re.sub(rf"\b{re.escape(name)}\s*\(", f"{name}{suffix}(", renamed)
        return renamed

    def _generate_harness_body(self, params: List[Tuple[str, str]], return_type: str) -> str:
        lines: List[str] = []
        lines.append("void equivalence_harness() {")
        lines.append("    struct SymbolicInput input;")
        lines.append("    __CPROVER_havoc_object(&input);")
        lines.append("    assume_legal_input(&input);")
        lines.append("")

        orig_args: List[str] = []
        opt_args: List[str] = []
        array_meta: List[Tuple[str, str, str, int]] = []

        for param_type, param_name in params:
            orig_name = f"{param_name}_orig"
            opt_name = f"{param_name}_opt"
            if "[" in param_type or "*" in param_type:
                lines.append(f"    {self._storage_decl(param_type, param_name, '_orig')};")
                lines.append(f"    {self._storage_decl(param_type, param_name, '_opt')};")
                element_type = self._element_type(param_type, param_name)
                extent = self._extent(param_type)
                lines.append(f"    memcpy({orig_name}, input.{param_name}, sizeof({orig_name}));")
                lines.append(f"    memcpy({opt_name}, input.{param_name}, sizeof({opt_name}));")
                array_meta.append((param_name, element_type, orig_name, extent))
                orig_args.append(orig_name)
                opt_args.append(opt_name)
            else:
                lines.append(f"    {param_type} {orig_name} = input.{param_name};")
                lines.append(f"    {param_type} {opt_name} = input.{param_name};")
                orig_args.append(orig_name)
                opt_args.append(opt_name)

        lines.append("")
        joined_orig_args = ", ".join(orig_args)
        joined_opt_args = ", ".join(opt_args)

        if return_type != "void":
            lines.append(f"    {return_type} result_orig = {self.function_name}_orig({joined_orig_args});")
            lines.append(f"    {return_type} result_opt = {self.function_name}_opt({joined_opt_args});")
            lines.append('    __CPROVER_assert(result_orig == result_opt, "return value mismatch");')
        else:
            lines.append(f"    {self.function_name}_orig({joined_orig_args});")
            lines.append(f"    {self.function_name}_opt({joined_opt_args});")

        for param_name, element_type, orig_name, extent in array_meta:
            opt_name = f"{param_name}_opt"
            lines.append(
                f'    __CPROVER_assert(memcmp({orig_name}, {opt_name}, sizeof({orig_name})) == 0, "{param_name} state mismatch");'
            )

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def generate(self) -> str:
        opt_code_raw = self.read_file(self.optimized_code_path)
        orig_code_raw = self.read_file(self.original_code_path)
        if not opt_code_raw or not orig_code_raw:
            return None

        opt_code, opt_defines = self.preprocess_code(opt_code_raw)
        orig_code, orig_defines = self.preprocess_code(orig_code_raw)
        all_defines = list(dict.fromkeys(opt_defines + orig_defines))

        opt_signature = self.parser.extract_function_signature(opt_code, self.function_name)
        orig_signature = self.parser.extract_function_signature(orig_code, self.function_name)
        if not opt_signature or not orig_signature:
            print(f"Error: Could not find function '{self.function_name}' in source files")
            return None

        self._load_define_values(all_defines)
        params = self.parser.parse_parameters(opt_signature)
        return_type = self.parser.extract_return_type(opt_signature)
        defines_section = "\n".join(all_defines) if all_defines else ""
        opt_code = self._rename_defined_functions(opt_code, "_opt")
        orig_code = self._rename_defined_functions(orig_code, "_orig")

        parts = [
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <string.h>",
            "",
            defines_section,
            "",
            "void __CPROVER_assume(_Bool);",
            "void __CPROVER_assert(_Bool, const char *);",
            "void __CPROVER_havoc_object(void *);",
            "",
            orig_code,
            "",
            opt_code,
            "",
            self._generate_input_struct(params),
            "",
            self._generate_input_assumptions(params).rstrip(),
            "",
            self._generate_harness_body(params, return_type).rstrip(),
            "",
        ]
        return "\n".join(part for part in parts if part is not None)

    def save(self, output_filename: str = "cbmc_harness.c") -> str:
        code = self.generate()
        if code is None:
            return None

        output_path = os.path.join(self.output_dir, output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"CBMC harness generated: {output_path}")
            return output_path
        except OSError as exc:
            print(f"Error writing output file: {exc}")
            return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CBMC equivalence harness for software-style C/C++ code.",
    )
    parser.add_argument("optimized_code", help="Path to optimized source file")
    parser.add_argument("original_code", help="Path to original source file")
    parser.add_argument("function_name", help="Function to compare")
    parser.add_argument("output_dir", nargs="?", default=".", help="Output directory")
    parser.add_argument("version", nargs="?", default=None, help="Optional version suffix")
    args = parser.parse_args()

    if args.version:
        output_filename = f"cbmc_harness_{args.version}.c"
    else:
        output_filename = "cbmc_harness.c"

    generator = CBMCHarnessGenerator(
        args.optimized_code,
        args.original_code,
        args.function_name,
        args.output_dir,
    )
    output_path = generator.save(output_filename)
    if output_path:
        print(f"Success: CBMC harness generated at {output_path}")
        sys.exit(0)

    print("Failed to generate CBMC harness")
    sys.exit(1)


if __name__ == "__main__":
    main()
