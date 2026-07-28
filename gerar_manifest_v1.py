#!/usr/bin/env python3
"""
Gera o manifest da versao 1.0 do dataset
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Em vez de duplicar as imagens numa pasta dataset_v1.0/ (o git ja versiona
o estado atual de sows_farm_02.v2i.yolo26/train,valid,test), este script
"congela" a v1.0 como um manifest.json: contagens por split, sha256 de
cada imagem (para detectar qualquer alteracao futura) e o resumo do
processamento ja aplicado (vazamento, qualidade, validacao estrutural).

O commit atual deve ser marcado com a tag git "dataset-v1.0" logo depois
de gerar este manifest.

Uso
    python gerar_manifest_v1.py sows_farm_02.v2i.yolo26
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

EXTENSOES_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def commit_atual() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "desconhecido"


def main():
    p = argparse.ArgumentParser(description="Gera manifest.json da v1.0 do dataset")
    p.add_argument("raiz", help="pasta do dataset (contem data.yaml)")
    p.add_argument("--saida", default="dataset_v1.0_manifest.json")
    args = p.parse_args()

    raiz = Path(args.raiz)
    data_yaml = raiz / "data.yaml"
    if not data_yaml.is_file():
        print("data.yaml nao encontrado em %s" % raiz)
        sys.exit(1)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    manifest = {
        "versao": "1.0",
        "data_geracao": date.today().isoformat(),
        "commit_git": commit_atual(),
        "classes": cfg["names"],
        "nc": cfg["nc"],
        "splits": {},
        "processamento_aplicado": [
            "Correcao de vazamento entre splits: 17 imagens quase-duplicadas "
            "(mesma cena em mais de um split) movidas para "
            "_quarentena_vazamento/ (ver log_correcao_vazamento.csv)",
            "Remocao de baixa qualidade: 58 imagens (borrada, subexposta, "
            "duplicata dentro do split, resolucao baixa) movidas para "
            "_quarentena_qualidade/ (ver log_remocao_qualidade.csv)",
            "Validacao estrutural dos rotulos: OK, 0 erros "
            "(validar_estrutura.py)",
        ],
        "pendencias": [
            "7 imagens com veredito 'duvidosa' permanecem no dataset ativo, "
            "sem revisao manual ainda (ver revisar_manualmente.csv)",
        ],
        "arquivos": [],
    }

    print("Calculando hash sha256 das imagens (pode levar um instante)...")
    for split in SPLITS:
        pasta_img = raiz / split / "images"
        pasta_lbl = raiz / split / "labels"
        imgs = sorted(f for f in pasta_img.iterdir() if f.suffix.lower() in EXTENSOES_IMG)

        n_objetos = 0
        for lbl in pasta_lbl.glob("*.txt"):
            n_objetos += sum(1 for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip())

        manifest["splits"][split] = {"imagens": len(imgs), "objetos": n_objetos}

        for img in imgs:
            manifest["arquivos"].append({
                "split": split,
                "arquivo": img.name,
                "sha256": sha256_arquivo(img),
            })

    saida = Path(args.saida)
    saida.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print("MANIFEST DA V1.0 GERADO")
    print("=" * 60)
    for split in SPLITS:
        s = manifest["splits"][split]
        print("  %-6s %5d imagens, %5d objetos" % (split, s["imagens"], s["objetos"]))
    print("  commit: %s" % manifest["commit_git"])
    print("  salvo em: %s" % saida.resolve())


if __name__ == "__main__":
    main()
