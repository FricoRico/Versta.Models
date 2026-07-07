import importlib
import pkgutil

from . import base as _base

_registry: dict[str, type[_base.BaseScraper]] = {}
_discovered = False


def register(name: str):
    """Decorator that registers a scraper class under the given name."""

    def decorator(cls: type[_base.BaseScraper]) -> type[_base.BaseScraper]:
        _registry[name] = cls
        return cls

    return decorator


def _discover_scrapers():
    """Import all scraper modules in this package so @register decorators fire."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    package = __package__ or __name__.rsplit(".", 1)[0]
    for _imp, modname, _ispkg in pkgutil.iter_modules(
        importlib.import_module(package).__path__
    ):
        if modname.startswith("_") or modname in ("base", "registry", "types", "pipeline", "__main__"):
            continue
        importlib.import_module(f".{modname}", package=package)


def get_scraper(name: str) -> _base.BaseScraper:
    """Instantiate a scraper by registered name."""
    _discover_scrapers()
    if name not in _registry:
        available = ", ".join(_registry)
        raise ValueError(
            f"Unknown scraper '{name}'. Available: {available}"
        )
    return _registry[name]()


def list_scrapers() -> list[str]:
    _discover_scrapers()
    return list(_registry.keys())
