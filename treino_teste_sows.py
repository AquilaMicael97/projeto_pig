#!/usr/bin/env python3
"""
Treino-teste + inferencia — deteccao de porca e leitao
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

O QUE ESTE SCRIPT FAZ
    1. Treina um YOLO nano por POUCAS epocas sobre o dataset indicado.
       Nao e o modelo final. E uma SONDA: serve para verificar se o dataset
       e treinavel e se as anotacoes fazem sentido.
    2. Roda o modelo resultante sobre algumas imagens de teste e salva os
       resultados com as caixas detectadas desenhadas.

O QUE ESPERAR
    Em CPU, mesmo poucas epocas levam algum tempo (minutos a dezenas de
    minutos, conforme o PC). Se ao final o modelo detectar leitoes e porca
    aproximadamente nos lugares certos, o dataset esta sao. Se nao detectar
    nada, ha problema na anotacao ou na estrutura de pastas.

INSTALACAO (rode uma vez, no terminal)
    pip install ultralytics

USO
    python treino_teste_sows.py "C:\\projetopig\\dataset\\sows_farm\\sows_farm_02.v2i.yolo26\\data.yaml"

    Opcoes:
    --epocas 5        numero de epocas (padrao 5; aumente se quiser)
    --imgsz 480       tamanho da imagem (padrao 480; menor = mais rapido)
    --modelo yolo11n.pt   modelo base (padrao yolo11n.pt)

OBSERVACAO SOBRE O MODELO BASE
    Uso yolo11n.pt como padrao por ser o mais estavel e amplamente testado
    hoje. Se voce ja tiver o yolo26n.pt disponivel, passe --modelo yolo26n.pt.
    Para este teste de sanidade, a diferenca entre os dois e irrelevante.
"""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Treino-teste e inferencia de deteccao de porca/leitao")
    p.add_argument("data_yaml", help="caminho do data.yaml do dataset")
    p.add_argument("--epocas", type=int, default=5,
                   help="numero de epocas de treino (padrao 5)")
    p.add_argument("--imgsz", type=int, default=480,
                   help="tamanho da imagem de entrada (padrao 480)")
    p.add_argument("--modelo", default="yolo11n.pt",
                   help="modelo base (padrao yolo11n.pt)")
    p.add_argument("--saida", default="resultado_treino_teste",
                   help="pasta de saida")
    args = p.parse_args()

    yaml_path = Path(args.data_yaml)
    if not yaml_path.is_file():
        print("data.yaml nao encontrado: %s" % yaml_path)
        print("Verifique o caminho. Ele deve apontar para o arquivo data.yaml,")
        print("nao para a pasta do dataset.")
        sys.exit(1)

    # Importa aqui para dar mensagem clara se o pacote nao estiver instalado
    try:
        from ultralytics import YOLO
    except ImportError:
        print("O pacote 'ultralytics' nao esta instalado.")
        print("Rode no terminal:  pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("TREINO-TESTE — sonda de sanidade do dataset")
    print("=" * 60)
    print("data.yaml : %s" % yaml_path)
    print("modelo    : %s" % args.modelo)
    print("epocas    : %d" % args.epocas)
    print("imgsz     : %d" % args.imgsz)
    print("dispositivo: CPU")
    print("=" * 60)
    print()

    modelo = YOLO(args.modelo)

    # ----- TREINO -----
    print(">> Iniciando treino de teste. Em CPU isso pode demorar.\n")
    modelo.train(
        data=str(yaml_path),
        epochs=args.epocas,
        imgsz=args.imgsz,
        device="cpu",
        batch=4,
        workers=2,
        project=args.saida,
        name="treino",
        exist_ok=True,
        # augmentation leve — este e so um teste de sanidade
        degrees=0.0,
        fliplr=0.5,
        mosaic=0.0,
        verbose=True,
    )

    # ----- INFERENCIA DE VERIFICACAO -----
    # Localiza a pasta de imagens de teste (ou validacao) a partir do yaml
    base = yaml_path.parent
    candidatos = [base / "test" / "images",
                  base / "valid" / "images",
                  base / "train" / "images"]
    pasta_img = next((c for c in candidatos if c.is_dir()), None)

    if pasta_img is None:
        print("\nTreino concluido, mas nao encontrei pasta de imagens para")
        print("inferencia. Verifique a estrutura train/valid/test.")
        return

    imgs = sorted(list(pasta_img.glob("*.jpg")) + list(pasta_img.glob("*.png")))[:6]
    if not imgs:
        print("\nNenhuma imagem encontrada em %s" % pasta_img)
        return

    print("\n>> Rodando inferencia em %d imagens de %s\n" % (len(imgs), pasta_img.name))

    saida_infer = Path(args.saida) / "deteccoes"
    saida_infer.mkdir(parents=True, exist_ok=True)

    melhor = Path(args.saida) / "treino" / "weights" / "best.pt"
    modelo_treinado = YOLO(str(melhor)) if melhor.exists() else modelo

    for img in imgs:
        res = modelo_treinado.predict(
            source=str(img), imgsz=args.imgsz, device="cpu",
            conf=0.05, save=True, project=str(saida_infer), name="pred", exist_ok=True,
            verbose=False)
        # resumo por imagem, com a confianca de cada deteccao
        r = res[0]
        nomes = r.names
        itens = []
        for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            itens.append("%s(%.0f%%)" % (nomes[int(c)], cf * 100))
        resumo = ", ".join(itens) if itens else "nada detectado"
        print("  %-50s -> %s" % (img.name[:50], resumo))

    print()
    print("=" * 60)
    print("CONCLUIDO")
    print("=" * 60)
    print("Metricas e curvas do treino: %s" % (Path(args.saida) / "treino"))
    print("Imagens com deteccoes:       %s" % saida_infer)
    print()
    print("Como ler o resultado:")
    print("  - Abra as imagens em 'deteccoes/pred' e compare as caixas")
    print("    detectadas com onde a porca e os leitoes realmente estao.")
    print("  - Os percentuais acima sao a CONFIANCA de cada deteccao. Com")
    print("    poucas epocas ela fica baixa (5-30%) — isso e NORMAL num teste")
    print("    curto. O que importa e a POSICAO: o modelo achou porca e leitao")
    print("    aproximadamente nos lugares certos? Se sim, o dataset e treinavel.")
    print("  - Se nao detectou NADA em imagem nenhuma nem com corte baixo, o")
    print("    problema esta na anotacao ou no caminho das pastas no data.yaml.")
    print("  - Para um modelo de verdade, rode com mais epocas (ex: --epocas 50).")


if __name__ == "__main__":
    main()