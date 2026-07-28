#!/usr/bin/env python3
"""
Validacao estatistica da distribuicao de classes entre splits
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Le os rotulos YOLO de train/valid/test e o data.yaml para saber os nomes
das classes, conta quantas instancias de cada classe existem em cada
split, calcula a razao de desbalanceamento e alerta sobre classes com
poucos exemplos em valid/test (risco de metrica de avaliacao instavel).

Uso
    python distribuicao_classes.py sows_farm_02.v2i.yolo26
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
MINIMO_RECOMENDADO_VALID_TEST = 20  # abaixo disso, metrica por classe fica instavel


def contar_split(pasta_labels: Path) -> Counter:
    contagem = Counter()
    for arq in pasta_labels.glob("*.txt"):
        for linha in arq.read_text(encoding="utf-8").strip().splitlines():
            partes = linha.split()
            if not partes:
                continue
            contagem[int(partes[0])] += 1
    return contagem


def main():
    p = argparse.ArgumentParser(description="Distribuicao de classes por split")
    p.add_argument("raiz", help="pasta do dataset (contem data.yaml)")
    p.add_argument("--saida", default="distribuicao_classes.csv")
    args = p.parse_args()

    raiz = Path(args.raiz)
    data_yaml = raiz / "data.yaml"
    if not data_yaml.is_file():
        print("data.yaml nao encontrado em %s" % raiz)
        sys.exit(1)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nomes = cfg["names"]

    contagens = {}
    for s in SPLITS:
        pasta = raiz / s / "labels"
        contagens[s] = contar_split(pasta) if pasta.is_dir() else Counter()

    linhas = []
    print("=" * 70)
    print("DISTRIBUICAO DE CLASSES POR SPLIT")
    print("=" * 70)
    print("%-22s %8s %8s %8s %10s" % ("classe", "train", "valid", "test", "total"))
    alertas = []
    totais_split = Counter()
    for idx, nome in enumerate(nomes):
        vals = {s: contagens[s].get(idx, 0) for s in SPLITS}
        total = sum(vals.values())
        for s in SPLITS:
            totais_split[s] += vals[s]
        print("%-22s %8d %8d %8d %10d" % (nome, vals["train"], vals["valid"], vals["test"], total))
        linhas.append({"classe": nome, "train": vals["train"], "valid": vals["valid"],
                        "test": vals["test"], "total": total})
        for s in ("valid", "test"):
            if vals[s] < MINIMO_RECOMENDADO_VALID_TEST:
                alertas.append("  %-22s %5s: apenas %d instancias (< %d recomendado)" %
                                (nome, s, vals[s], MINIMO_RECOMENDADO_VALID_TEST))

    print("-" * 70)
    print("%-22s %8d %8d %8d %10d" % ("TOTAL", totais_split["train"], totais_split["valid"],
                                       totais_split["test"], sum(totais_split.values())))

    maior = max(l["total"] for l in linhas)
    menor = min(l["total"] for l in linhas)
    print()
    print("Razao de desbalanceamento (classe mais comum / classe mais rara): %.1fx" %
          (maior / menor if menor else float("inf")))

    if alertas:
        print()
        print("ALERTAS (classes com poucas instancias em valid/test):")
        for a in alertas:
            print(a)

    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["classe", "train", "valid", "test", "total"])
        w.writeheader()
        w.writerows(linhas)
    print()
    print("CSV salvo em: %s" % args.saida)


if __name__ == "__main__":
    main()
