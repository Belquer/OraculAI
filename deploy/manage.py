"""Manage helper for deploy context."""

from __future__ import annotations

import click


@click.group()
def cli():
    """Management CLI"""


@cli.command()
def noop():
    """No-op command to satisfy Dockerfile copy"""
    click.echo("noop")


if __name__ == "__main__":
    cli()
