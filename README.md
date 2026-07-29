# Sistema Anti-Esmagamento de Leitões

Pipeline de preparação e validação de dataset para um modelo de visão computacional (YOLO) que detecta porcas e leitões em baias de maternidade, com o objetivo de identificar situações de risco de esmagamento do leitão pela porca.

## Dataset

- **Fonte:** [Roboflow — sows-farm-02](https://universe.roboflow.com/btg-4kqp7/sows-farm-02) (licença CC BY 4.0)
- **Formato:** YOLO (uma caixa por linha: `classe centro_x centro_y largura altura`, normalizados)
- **Classes (6):** `piglet`, `sow-sit`, `sow-sleep`, `sow-sleep-lactate`, `sow-stand`, `sow-stand-feed`
- **Versão atual: v1.0** (tag git `dataset-v1.0`) — 1097 imagens originais, já com vazamento entre splits corrigido e imagens de baixa qualidade removidas

| Split | Imagens |
|---|---|
| train | 789 |
| valid | 129 |
| test  | 104 |

Detalhes completos da limpeza e da distribuição de classes: [RELATORIO_QUALIDADE.md](RELATORIO_QUALIDADE.md).

## Pipeline de pré-processamento e validação

Os scripts abaixo rodam nessa ordem — cada um consome a saída do anterior:

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `triagem_qualidade_dataset.py` | Avalia nitidez, brilho, contraste, resolução e duplicatas visuais (hash perceptual) de cada imagem. Gera `relatorio_qualidade.csv` / `resumo_qualidade.txt`. |
| 2 | `corrigir_vazamento.py` | Remove (move para quarentena) imagens quase-duplicadas que cruzam os splits train/valid/test, evitando contaminação da avaliação. |
| 3 | `remover_baixa_qualidade.py` | Move para quarentena as imagens com veredito "descartar" da triagem. As "duvidosa" ficam para revisão manual (`revisar_manualmente.csv`). |
| 4 | `validar_estrutura.py` | Gate de integridade: confere pareamento imagem↔rótulo, `class_id` válido e coordenadas de bounding box corretas. Falha (`exit 1`) se achar erro estrutural. |
| 5 | `distribuicao_classes.py` | Conta instâncias por classe e por split, calcula a razão de desbalanceamento e alerta classes com poucos exemplos em valid/test. |
| 6 | `gerar_manifest_v1.py` | Gera `dataset_v1.0_manifest.json` com contagens, hash sha256 de cada imagem e o resumo do processamento aplicado — o "congelamento" da versão v1.0. |

Scripts complementares:

- `gerar_painel_augmentation.py` — gera um painel visual comparando o efeito de cada transformação de augmentation numa imagem real (usado para decidir quais técnicas fazem sentido para esta cena).
- `treino_teste_sows.py` — treina um YOLO (nano) com a estratégia de augmentation definida e roda inferência de verificação. Detecta GPU automaticamente (`--device auto`, padrão).

## Principais achados

- **Vazamento entre splits:** 17 imagens quase-duplicadas apareciam simultaneamente em train e valid/test (mesma câmera, 1–2 min de diferença) — corrigido antes de qualquer treino.
- **Desbalanceamento de classes:** razão de ~51x entre a classe mais comum (`piglet`) e a mais rara (`sow-sleep`) em número de instâncias. Em área ocupada no quadro, o leitão ocupa em média ~2% do frame contra 41–57% das classes de porca.
- **Qualidade de imagem:** 94% do dataset aproveitável sem ajustes; 58 imagens removidas por defeito grave (borrada, subexposta, duplicata, baixa resolução); 0 imagens corrompidas.

## Como usar

Instalação (uma vez):
```bash
pip install ultralytics albumentations opencv-python numpy pillow pyyaml
```

Rodar o pipeline completo, na ordem:
```bash
python triagem_qualidade_dataset.py sows_farm_02.v2i.yolo26
python corrigir_vazamento.py
python remover_baixa_qualidade.py
python validar_estrutura.py sows_farm_02.v2i.yolo26
python distribuicao_classes.py sows_farm_02.v2i.yolo26
python gerar_manifest_v1.py sows_farm_02.v2i.yolo26
```

Treinar (detecta GPU automaticamente; usa CPU se não houver):
```bash
python treino_teste_sows.py sows_farm_02.v2i.yolo26/data.yaml --epocas 50
```

## Status (Sprint 3 — Preparação dos Dados e Estruturação do Pipeline)

- [x] Estrutura organizada (train/valid/test)
- [x] Limpeza: duplicadas, vazamento entre splits e baixa qualidade removidos
- [x] Pipeline automatizado de pré-processamento e validação
- [x] Validação estatística de distribuição e balanceamento das classes
- [x] Estratégia de Data Augmentation configurada
- [x] Dataset v1.0 gerado e versionado (tag `dataset-v1.0` + manifest)
- [x] Relatório de qualidade e distribuição das classes
- [ ] Revisão manual das 7 imagens "duvidosa" (`revisar_manualmente.csv`)
- [ ] Anotação/revisão completa conforme classes da Sprint 2
- [ ] Treinamento piloto em GPU
