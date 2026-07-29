#!/usr/bin/env python3
"""
Painel de revisao manual das imagens "duvidosa"
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Le revisar_manualmente.csv e monta um contact-sheet com cada imagem e suas
caixas YOLO desenhadas, pra decisao visual rapida (ficar ou sair do
dataset v1.0).

Uso
    python revisar_duvidosas.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELULA = 300
MARGEM = 14
RODAPE = 46
COLUNAS = 4


def ler_imagem(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def achar_rotulo(img_path: Path):
    alt = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    return alt if alt.exists() else None


def ler_caixas(rotulo, largura, altura):
    caixas = []
    if rotulo is None:
        return caixas
    for linha in rotulo.read_text(encoding="utf-8").strip().splitlines():
        p = linha.split()
        if len(p) < 5:
            continue
        _, xc, yc, w, h = p[0], *map(float, p[1:5])
        cx, cy, bw, bh = xc * largura, yc * altura, w * largura, h * altura
        caixas.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return caixas


def desenhar(img, caixas):
    saida = img.copy()
    alt, larg = saida.shape[:2]
    for (x1, y1, x2, y2) in caixas:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(larg - 1, x2)), int(min(alt - 1, y2))
        cv2.rectangle(saida, (x1, y1), (x2, y2), (0, 210, 255), 2)
    return saida


def ajustar(img):
    alt, larg = img.shape[:2]
    escala = min(CELULA / larg, CELULA / alt)
    nova = cv2.resize(img, (max(1, int(larg * escala)), max(1, int(alt * escala))))
    tela = np.full((CELULA, CELULA, 3), 255, dtype=np.uint8)
    y, x = (CELULA - nova.shape[0]) // 2, (CELULA - nova.shape[1]) // 2
    tela[y:y + nova.shape[0], x:x + nova.shape[1]] = nova
    return tela


def carregar_fonte(tam):
    for c in ("C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(c, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    linhas = list(csv.DictReader(open("revisar_manualmente.csv", encoding="utf-8")))
    nitidez_por_arquivo = {}
    with open("relatorio_qualidade.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nitidez_por_arquivo[r["arquivo"]] = r["nitidez"]

    n = len(linhas)
    lin_grade = (n + COLUNAS - 1) // COLUNAS
    largura = COLUNAS * CELULA + (COLUNAS + 1) * MARGEM
    altura = lin_grade * (CELULA + RODAPE) + (lin_grade + 1) * MARGEM

    tela = Image.new("RGB", (largura, altura), (255, 255, 255))
    desenho = ImageDraw.Draw(tela)
    f_tit = carregar_fonte(13)

    for i, r in enumerate(linhas):
        img_path = Path(r["caminho"])
        img = ler_imagem(img_path)
        if img is None:
            continue
        alt, larg = img.shape[:2]
        rotulo = achar_rotulo(img_path)
        caixas = ler_caixas(rotulo, larg, alt)
        com_caixas = desenhar(img, caixas)
        celula = ajustar(com_caixas)

        col, lin = i % COLUNAS, i // COLUNAS
        x = MARGEM + col * (CELULA + MARGEM)
        y = MARGEM + lin * (CELULA + RODAPE + MARGEM)
        tela.paste(Image.fromarray(cv2.cvtColor(celula, cv2.COLOR_BGR2RGB)), (x, y))
        desenho.rectangle([x, y, x + CELULA - 1, y + CELULA - 1], outline=(200, 200, 200))

        split = "train" if "train" in str(img_path) else ("valid" if "valid" in str(img_path) else "test")
        nit = nitidez_por_arquivo.get(r["arquivo"], "?")
        desenho.text((x, y + CELULA + 4), "%s | nitidez=%s | %d caixas" % (split, nit, len(caixas)),
                     font=f_tit, fill=(30, 30, 30))

    saida = Path("revisao_anotacoes") / "revisao_duvidosas.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    tela.save(saida)
    print("Painel salvo em: %s" % saida.resolve())


if __name__ == "__main__":
    main()
