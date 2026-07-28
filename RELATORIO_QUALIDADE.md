# Relatório de Qualidade e Distribuição das Classes

Dataset: `sows_farm_02.v2i.yolo26` — sistema anti-esmagamento de leitões em baia de maternidade
Sprint 3 — entregável "Relatório de qualidade e distribuição das classes"

## 1. Visão geral

| Split | Original (Roboflow) | Após vazamento | Após remoção de baixa qualidade |
|---|---|---|---|
| train | 848 | 833 | 789 |
| valid | 142 | 140 | 129 |
| test  | 107 | 107 | 104 |
| **Total** | **1097** | **1080** | **1022** |

## 2. Correção de vazamento entre splits

A checagem de duplicatas por hash perceptual (dHash) rodada sobre o dataset inteiro encontrou 46 duplicatas, das quais **17 cruzavam splits** — a mesma cena (frames da mesma câmera com 1–2 min de diferença) aparecia simultaneamente em train e valid/test, contaminando a avaliação.

Política aplicada: ao colidir train×valid/test, remove-se o lado train; ao colidir valid×test, remove-se o lado valid (test fica intocado, por ser o benchmark final). As 17 imagens + rótulos foram movidos para `sows_farm_02.v2i.yolo26/_quarentena_vazamento/` (não apagados). Detalhe por arquivo em `log_correcao_vazamento.csv`.

## 3. Triagem e remoção de qualidade de imagem

Critérios: nitidez (variância do Laplaciano), brilho médio, contraste (desvio-padrão), resolução mínima (320px) e duplicata visual dentro do mesmo split (dHash, tolerância 1 bit). Análise completa em `relatorio_qualidade.csv` / `resumo_qualidade.txt` (gerada sobre as 1080 imagens pós-vazamento).

| Veredito | Imagens | % | Ação |
|---|---|---|---|
| Aproveitável | 1015 | 94,0% | mantidas |
| Duvidosa | 7 | 0,6% | **mantidas, pendente revisão manual** |
| Descartar | 58 | 5,4% | **removidas** |

Motivos das 58 removidas (uma imagem pode ter mais de um): duplicata dentro do split, borrada, subexposta, nitidez limítrofe, baixo contraste. Movidas (não apagadas) para `sows_farm_02.v2i.yolo26/_quarentena_qualidade/`, log em `log_remocao_qualidade.csv`.

As **7 "duvidosa"** continuam no dataset ativo e estão listadas em `revisar_manualmente.csv` para decisão caso a caso (ficar ou sair) antes de fechar a v1.0.

Nenhuma imagem corrompida foi encontrada (0 arquivos ilegíveis) em nenhuma das duas rodadas.

## 4. Validação estrutural dos rótulos

Checagem de integridade (não de qualidade): pareamento imagem↔rótulo, `class_id` dentro de 0–5, coordenadas normalizadas válidas, sem caixa degenerada. Script: `validar_estrutura.py`.

**Resultado: OK — nenhum erro estrutural encontrado** em nenhum split, sobre o dataset já limpo (789/129/104 imagens, 4119/731/561 objetos válidos, 0 rótulos vazios, 0 órfãos).

## 5. Distribuição e balanceamento das classes

Contagem de instâncias (caixas) no dataset atual, já sem o vazamento e sem as 58 imagens de baixa qualidade. Detalhe em `distribuicao_classes.csv`.

| Classe | train | valid | test | total |
|---|---|---|---|---|
| piglet | 3330 | 602 | 457 | 4389 |
| sow-sleep-lactate | 441 | 90 | 58 | 589 |
| sow-sit | 129 | 15 | 12 | 156 |
| sow-stand-feed | 78 | 9 | 8 | 95 |
| sow-stand | 77 | 10 | 10 | 97 |
| sow-sleep | 64 | 5 | 16 | 85 |
| **Total** | **4119** | **731** | **561** | **5411** |

**Razão de desbalanceamento (classe mais comum / mais rara): 51,6x** (`piglet` vs `sow-sleep`) — subiu em relação à medição anterior (48,3x) porque a remoção de baixa qualidade tirou proporcionalmente mais instâncias de classes raras.

**Alertas** — classes abaixo de 20 instâncias em valid e/ou test:
- `sow-sit`: valid 15, test 12
- `sow-sleep`: **valid 5** (caiu de 12 para 5 — ponto de atenção, ficou bem abaixo do mínimo), test 16
- `sow-stand`: valid 10, test 10
- `sow-stand-feed`: valid 9, test 8

## 6. Observações e recomendações

- O desbalanceamento (51,6x) é o maior risco de qualidade do dataset — `piglet` domina; classes de comportamento da porca têm poucos exemplos, e `sow-sleep` em valid ficou criticamente baixo (5 instâncias) após a limpeza.
- Decisão para esta sprint: sem oversampling/augmentation direcionado agora — medir o efeito real do desbalanceamento no treinamento piloto antes de agir.
- Pendência: as 7 imagens "duvidosa" ainda precisam de revisão manual (`revisar_manualmente.csv`) antes de fechar o `dataset_v1.0/`.
- Pipeline de validação automatizada tem 3 estágios rodando e passando: qualidade de imagem, vazamento entre splits, integridade estrutural do rótulo.

## Arquivos gerados

- `relatorio_qualidade.csv`, `resumo_qualidade.txt` — triagem de qualidade por imagem
- `log_correcao_vazamento.csv` — imagens removidas por vazamento entre splits
- `log_remocao_qualidade.csv` — imagens removidas por baixa qualidade
- `revisar_manualmente.csv` — imagens "duvidosa" pendentes de revisão humana
- `distribuicao_classes.csv` — contagem de instâncias por classe/split
- `validar_estrutura.py` — validação estrutural dos rótulos (gate de integridade)
