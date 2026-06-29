"""Rich terminal renderer — prints a formatted game snapshot after every step."""

from __future__ import annotations

from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

SUIT_COLOR = {"♠": "white", "♣": "white", "♥": "bright_red", "♦": "bright_red"}

PHASE_LABELS: dict[str, tuple[str, str]] = {
    "blind_select":    ("BLIND SELECT",   "yellow"),
    "selecting_hand":  ("PLAYING",        "green"),
    "round_eval":      ("ROUND COMPLETE", "cyan"),
    "shop":            ("SHOP",           "magenta"),
    "pack_opening":    ("PACK OPENING",   "blue"),
    "game_over":       ("GAME OVER",      "red"),
}


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------


def _card_text(card: dict) -> Text:
    if card.get("type") == "playing":
        symbol = card["suit_symbol"]
        color = SUIT_COLOR.get(symbol, "white")
        label = f" {card['rank_short']}{symbol} "
        t = Text(label, style=f"bold {color} on grey15")
        if card.get("debuff"):
            t.stylize("strike dim")
        return t
    else:
        name = (card.get("name") or card.get("key", "?"))[:14]
        return Text(f" {name} ", style="bold yellow on grey15")


def _hand_renderable(hand: list[dict]) -> Text:
    if not hand:
        return Text("  (empty)", style="dim")
    row = Text()
    for i, card in enumerate(hand):
        if i:
            row.append("  ")
        row.append_text(_card_text(card))
    return row


def _joker_table(jokers: list[dict]) -> Table | Text:
    if not jokers:
        return Text("  (none)", style="dim")
    t = Table(box=None, padding=(0, 1), show_header=False)
    t.add_column("name", style="bold yellow", no_wrap=True)
    t.add_column("desc", style="dim", overflow="fold")
    for j in jokers:
        desc = j.get("desc", "")
        if len(desc) > 48:
            desc = desc[:45] + "…"
        badge = ""
        if j.get("eternal"):
            badge += " [cyan]∞[/cyan]"
        if j.get("perishable"):
            badge += " [red]🍂[/red]"
        if j.get("rental"):
            badge += " [yellow]💰[/yellow]"
        t.add_row(Text(f"{j['name']}{badge}"), Text(desc))
    return t


def _score_bar(chips: int, blind_chips: int | None) -> Text:
    if not blind_chips:
        return Text(f"  {chips:,} chips", style="cyan")
    pct = min(chips / blind_chips, 1.0)
    width = 28
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct >= 1 else ("yellow" if pct >= 0.5 else "cyan")
    t = Text()
    t.append(f"  {bar}", style=color)
    t.append(f"  {chips:,}", style="bold")
    t.append(" / ", style="dim")
    t.append(f"{blind_chips:,}", style="dim")
    t.append(f"  ({pct:.0%})", style="dim")
    return t


# ---------------------------------------------------------------------------
# RichRenderer
# ---------------------------------------------------------------------------


class RichRenderer:
    def render(self, snap: dict[str, Any]) -> None:
        phase_raw = snap.get("phase", "")
        phase_label, phase_color = PHASE_LABELS.get(phase_raw, (phase_raw.upper(), "white"))

        ante = snap.get("ante", 1)
        blind_on = snap.get("blind_on_deck", "")
        blind_name = snap.get("blind_name") or blind_on
        dollars = snap.get("dollars", 0)
        won = snap.get("won", False)

        # ── Header ──────────────────────────────────────────────────────
        header = Text()
        header.append(f" {phase_label} ", style=f"bold {phase_color} on grey15")
        header.append("  ")
        header.append(f"Ante {ante}  ·  {blind_name}", style="bold")
        header.append("   ")
        header.append(f"${dollars}", style="bold green")
        if won:
            header.append("  🏆 WON", style="bold yellow")

        console.rule(header, style="grey30")

        # ── Score progress ───────────────────────────────────────────────
        chips = snap.get("chips", 0)
        blind_chips = snap.get("blind_chips")
        score_text = _score_bar(chips, blind_chips)

        last_hand = snap.get("last_hand")
        hand_line = Text()
        if last_hand and last_hand.get("name"):
            hand_line.append(f"  {last_hand['name']}", style="bold white")
            hand_line.append("  ·  ", style="dim")
            hand_line.append(f"{last_hand['chips']} chips", style="cyan")
            hand_line.append(" × ", style="dim")
            hand_line.append(f"{last_hand['mult']} mult", style="bright_red")
            if snap.get("last_score"):
                hand_line.append(f"  =  {snap['last_score']:,}", style="bold yellow")

        # ── Hand ─────────────────────────────────────────────────────────
        hands_left = snap.get("hands_left", 0)
        discards_left = snap.get("discards_left", 0)
        hand_header = f"HAND  ·  Hands: {hands_left}  Discards: {discards_left}  ·  Deck: {snap.get('deck_remaining', 0)}"

        hand_panel = Panel(
            _hand_renderable(snap.get("hand", [])),
            title=f"[bold]{hand_header}[/bold]",
            border_style="green",
            padding=(0, 1),
        )

        # ── Jokers ───────────────────────────────────────────────────────
        joker_panel = Panel(
            _joker_table(snap.get("jokers", [])),
            title="[bold yellow]JOKERS[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )

        # ── Score panel ──────────────────────────────────────────────────
        score_content = Text()
        score_content.append_text(score_text)
        if hand_line.plain:
            score_content.append("\n")
            score_content.append_text(hand_line)

        score_panel = Panel(
            score_content,
            title="[bold cyan]SCORE[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

        console.print(score_panel)
        console.print(Columns([hand_panel, joker_panel], equal=False, expand=True))

        # ── Phase-specific extras ────────────────────────────────────────
        if phase_raw == "shop":
            self._render_shop(snap)
        elif phase_raw == "pack_opening":
            self._render_pack(snap)
        elif phase_raw == "blind_select":
            self._render_blind_select(snap)
        elif phase_raw == "round_eval":
            console.print(Panel(
                Text("  Earnings ready — Cash Out to continue.", style="bold green"),
                title="[bold]ROUND COMPLETE[/bold]",
                border_style="green",
            ))
        elif phase_raw == "game_over":
            label = "🏆  YOU WON THE RUN!" if won else "💀  GAME OVER"
            console.print(Panel(Text(f"  {label}", style="bold"), border_style="red"))

        # ── Last action ──────────────────────────────────────────────────
        last = snap.get("last_action")
        if last:
            console.print(Text(f"  → {last}", style="dim italic"))

        console.print()

    # ------------------------------------------------------------------

    def _render_shop(self, snap: dict) -> None:
        cards = snap.get("shop_cards", [])
        vouchers = snap.get("shop_vouchers", [])
        boosters = snap.get("shop_boosters", [])

        t = Table(title="SHOP", box=None, padding=(0, 1), show_header=True, border_style="magenta")
        t.add_column("Type", style="dim", width=10)
        t.add_column("Name", style="bold")
        t.add_column("Desc", style="dim", overflow="fold", max_width=50)
        t.add_column("$", style="green", width=4)

        for c in cards:
            t.add_row(c.get("set", "Card"), c.get("name", "?"), c.get("desc", "")[:48], str(c.get("cost", 0)))
        for v in vouchers:
            t.add_row("Voucher", v.get("name", "?"), v.get("desc", "")[:48], str(v.get("cost", 0)))
        for b in boosters:
            t.add_row("Pack", b.get("name", "?"), b.get("desc", "")[:48], str(b.get("cost", 0)))

        if cards or vouchers or boosters:
            console.print(Panel(t, border_style="magenta", padding=(0, 1)))

    def _render_pack(self, snap: dict) -> None:
        cards = snap.get("pack_cards", [])
        remaining = snap.get("pack_choices_remaining", 0)
        pack_type = snap.get("pack_type", "Pack")
        row = Text()
        for i, c in enumerate(cards):
            if i:
                row.append("  ")
            row.append_text(_card_text(c))
        console.print(Panel(
            row,
            title=f"[bold blue]{pack_type}  —  {remaining} pick(s) remaining[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        ))

    def _render_blind_select(self, snap: dict) -> None:
        blind_on = snap.get("blind_on_deck", "Small")
        blind_chips = snap.get("blind_chips")
        reroll_cost = snap.get("reroll_cost", 5)
        msg = Text()
        msg.append(f"  {blind_on} Blind", style="bold yellow")
        if blind_chips:
            msg.append(f"  —  Target: {blind_chips:,} chips", style="cyan")
        msg.append(f"  —  Reroll: ${reroll_cost}", style="dim")
        console.print(Panel(msg, title="[bold]SELECT BLIND[/bold]", border_style="yellow", padding=(0, 1)))
