import yaml
import sys
import os
import re
import argparse


class ConfigLoader:
    def __init__(self, default_path="configs/default.yaml", overrides=None):
        with open(default_path, "r") as f:
            file_map = yaml.safe_load(f)

        base_dir = os.path.dirname(default_path)
        self.config = {}

        for key, rel_path in file_map.items():
            full_path = os.path.join(base_dir, rel_path)
            with open(full_path, "r") as f:
                self.config[key] = yaml.safe_load(f) or {}

        if overrides:
            self._apply_overrides(overrides)

        self.config = self._resolve_references(self.config, self.config)

    @classmethod
    def from_cli(cls, default_path="configs/default.yaml", cli_args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
        args = parser.parse_args(args=cli_args)

        overrides = {}
        if args.set:
            for pair in args.set:
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid format: {pair}, expected key=value")
                key, value = pair.split("=", 1)
                overrides[key] = value

        return cls(default_path=default_path, overrides=overrides)

    def _apply_overrides(self, overrides):
        for k, v in overrides.items():
            path = k.split(".")
            cfg = self.config
            for key in path[:-1]:
                cfg = cfg.setdefault(key, {})
            cfg[path[-1]] = self._cast_value(v)

    def _cast_value(self, v):
        if isinstance(v, str):
            if v.lower() in ("true", "false"):
                return v.lower() == "true"
            if v.isdigit():
                return int(v)
            try:
                return float(v)
            except ValueError:
                pass
            if v.startswith("[") and v.endswith("]"):
                try:
                    return eval(v)
                except:
                    return v
        return v

    def _resolve_references(self, config, root_config):
        if isinstance(config, dict):
            return {k: self._resolve_references(v, root_config) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_references(i, root_config) for i in config]
        elif isinstance(config, str):
            matches = re.findall(r"\$\{([^}]+)\}", config)
            for match in matches:
                ref_path = match.split(".")
                ref_value = root_config
                try:
                    for ref in ref_path:
                        if not isinstance(ref_value, dict):
                            raise KeyError
                        ref_value = ref_value[ref]
                except (KeyError, TypeError):
                    raise ValueError(f"Reference not found: ${{{match}}}")
                config = config.replace(f"${{{match}}}", str(ref_value))
            return config
        else:
            return config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def as_dict(self):
        return self.config
