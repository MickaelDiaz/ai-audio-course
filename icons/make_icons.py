#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio AI Atlas — generateur d'icones « Studio Mono ».
Squircle noir chaud plein-cadre (maskable-safe) + marque corail :
un petit spectre de barres capsule, avec un leger halo corail.
Rendu en supersampling puis downscale LANCZOS pour des bords nets.
"""
from PIL import Image, ImageDraw, ImageFilter

SS = 2048                      # canvas master (supersample)
BG_TOP = (23, 23, 27)          # #17171b
BG_BOT = (10, 10, 12)          # #0a0a0c
CORAL_TOP = (255, 111, 83)     # #ff6f53
CORAL_BOT = (243, 64, 50)      # #f33240
HEIGHTS = [0.30, 0.52, 0.78, 1.0, 0.78, 0.52, 0.30]

def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def vgrad(size, top, bot):
    """Degrade vertical plein-cadre."""
    img = Image.new("RGB", (size, size), bot)
    px = img.load()
    for y in range(size):
        c = lerp(top, bot, y / (size - 1))
        for x in range(size):
            px[x, y] = c
    return img

def make_master():
    S = SS
    base = vgrad(S, BG_TOP, BG_BOT).convert("RGBA")

    # halo blanc tres doux en haut (profondeur « verre »)
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([S * 0.10, -S * 0.42, S * 0.90, S * 0.30],
               fill=(255, 255, 255, 26))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.05))
    base = Image.alpha_composite(base, glow)

    # geometrie des barres
    n = len(HEIGHTS)
    Wm = 0.50 * S                       # largeur totale de la marque
    gap_ratio = 0.6
    bar_w = Wm / (n + (n - 1) * gap_ratio)
    gap = bar_w * gap_ratio
    max_h = 0.46 * S
    mid_y = S * 0.50
    x0 = (S - Wm) / 2.0
    r = bar_w / 2.0

    bars = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bars)
    for i, hf in enumerate(HEIGHTS):
        h = max_h * hf
        cx = x0 + i * (bar_w + gap)
        top = mid_y - h / 2.0
        bot = mid_y + h / 2.0
        # degrade vertical par barre (subtil relief)
        col = lerp(CORAL_TOP, CORAL_BOT, hf * 0.5)
        bd.rounded_rectangle([cx, top, cx + bar_w, bot], radius=r, fill=col + (255,))

    # halo corail derriere les barres
    halo = bars.filter(ImageFilter.GaussianBlur(S * 0.018))
    halo = Image.eval(halo, lambda a: a)  # garde l'alpha
    base = Image.alpha_composite(base, halo)
    base = Image.alpha_composite(base, bars)
    return base

def main():
    master = make_master()
    for name, size in [("icon-512.png", 512), ("icon-192.png", 192),
                       ("apple-touch-icon.png", 180)]:
        out = master.resize((size, size), Image.LANCZOS).convert("RGB")
        out.save(name, "PNG")
        print("wrote", name, size)

if __name__ == "__main__":
    main()
