#!/usr/bin/env python3
"""
Remocao das imagens de baixa qualidade (veredito "descartar") do dataset
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Le relatorio_qualidade.csv (gerado por triagem_qualidade_dataset.py) e move
para quarentena as imagens com veredito "descartar" (motivos graves: borrada,
subexposta, superexposta, resolucao_baixa, duplicata dentro do mesmo split).

As imagens "duvidosa" NAO sao tocadas aqui - sao casos limitrofes que exigem
revisao manual antes de qualquer decisao. Ficam listadas em
revisar_manualmente.csv para essa revisao.

Nada e apagado: imagem + rotulo vao para
sows_farm_02.v2i.yolo26/_quarentena_qualidade/<split>/{images,labels}/

Uso
    python remover_baixa_qualidade.py
    python remover_baixa_qualidade.py --csv relatorio_qualidade.csv --raiz sows_farm_02.v2i.yolo26
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")


def split_de(caminho: str) -> str:
    partes = Path(caminho).parts
    for s in SPLITS:
        if s in partes:
            return s
    return "?"


def rotulo_de(caminho_imagem: Path) -> Path:
    return caminho_imagem.parent.parent / "labels" / (caminho_imagem.stem + ".txt")


def main():
    p = argparse.ArgumentParser(description="Remove (para quarentena) imagens de baixa qualidade")
    p.add_argument("--csv", default="relatorio_qualidade.csv")
    p.add_argument("--raiz", default="sows_farm_02.v2i.yolo26")
    p.add_argument("--quarentena", default=None,
                   help="pasta de quarentena (padrao: <raiz>/_quarentena_qualidade)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    raiz = Path(args.raiz)
    if not csv_path.is_file():
        print("CSV nao encontrado: %s" % csv_path)
        sys.exit(1)

    quarentena = Path(args.quarentena) if args.quarentena else raiz / "_quarentena_qualidade"

    linhas = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    descartar = [r for r in linhas if r["veredito"] == "descartar"]
    duvidosa = [r for r in linhas if r["veredito"] == "duvidosa"]

    log, movidas, faltando = [], 0, []
    for r in descartar:
        img_path = Path(r["caminho"])
        if not img_path.is_file():
            faltando.append(str(img_path))
            continue
        split_origem = split_de(r["caminho"])
        rot_path = rotulo_de(img_path)

        destino_img = quarentena / split_origem / "images" / img_path.name
        destino_rot = quarentena / split_origem / "labels" / rot_path.name
        destino_img.parent.mkdir(parents=True, exist_ok=True)
        destino_rot.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(img_path), str(destino_img))
        if rot_path.is_file():
            shutil.move(str(rot_path), str(destino_rot))
        movidas += 1
        log.append({"arquivo": r["arquivo"], "split_origem": split_origem,
                    "motivos": r["motivos"]})

    log_path = Path("log_remocao_qualidade.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "split_origem", "motivos"])
        w.writeheader()
        w.writerows(log)

    revisar_path = Path("revisar_manualmente.csv")
    with open(revisar_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "veredito", "motivos", "caminho"])
        w.writeheader()
        for r in duvidosa:
            w.writerow({"arquivo": r["arquivo"], "veredito": r["veredito"],
                       "motivos": r["motivos"], "caminho": r["caminho"]})

    print("=" * 60)
    print("REMOCAO DE IMAGENS DE BAIXA QUALIDADE")
    print("=" * 60)
    print("Descartar (removidas para quarentena): %d" % movidas)
    if faltando:
        print("Ja nao existiam: %d" % len(faltando))
    print("Duvidosa (NAO tocadas, para revisao manual): %d" % len(duvidosa))
    print()
    print("Quarentena: %s" % quarentena.resolve())
    print("Log:        %s" % log_path.resolve())
    print("Revisar:    %s" % revisar_path.resolve())

    print()
    print("Contagem de imagens por split apos a remocao:")
    for s in SPLITS:
        pasta = raiz / s / "images"
        n = len(list(pasta.glob("*"))) if pasta.is_dir() else 0
        print("  %-6s %d" % (s, n))


if __name__ == "__main__":
    main()
