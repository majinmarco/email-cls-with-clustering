__all__ = ["main", "setup_tracking", "init_talon", "parse_email", "expand_emails"]


def __getattr__(name: str):
    if name == "setup_tracking":
        from email_cls_with_clustering.tracking import setup_tracking

        return setup_tracking
    if name == "init_talon":
        from email_cls_with_clustering.talon_compat import init_talon

        return init_talon
    if name in {"parse_email", "expand_emails"}:
        from email_cls_with_clustering import preprocess

        return getattr(preprocess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    print("Hello from email-cls-with-clustering!")
