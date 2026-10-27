import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Tutor Estatístico",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Tutor Estatístico")
st.caption(
    "Exploração descritiva dos dados e orientação educativa para escolha de testes estatísticos."
)

st.info(
    "Este aplicativo tem finalidade educativa. A sugestão de um teste não substitui "
    "a avaliação do desenho do estudo, dos pressupostos estatísticos e do contexto científico."
)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def carregar_csv(arquivo):
    """Tenta ler CSV detectando automaticamente o separador."""
    tentativas = [
        dict(sep=None, engine="python"),
        dict(sep=";", decimal=","),
        dict(sep=","),
        dict(sep="\t"),
    ]

    ultimo_erro = None

    for kwargs in tentativas:
        try:
            arquivo.seek(0)
            df = pd.read_csv(arquivo, **kwargs)
            if df.shape[1] >= 1:
                return df
        except Exception as erro:
            ultimo_erro = erro

    raise ultimo_erro


def classificar_coluna(serie):
    """Classificação inicial apenas para auxiliar a interface."""
    s = serie.dropna()

    if len(s) == 0:
        return "Sem dados"

    n_unicos = s.nunique()

    if pd.api.types.is_bool_dtype(s):
        return "Categórica binária"

    if pd.api.types.is_numeric_dtype(s):
        if n_unicos == 2:
            return "Numérica binária"
        if pd.api.types.is_integer_dtype(s) and n_unicos <= 10:
            return "Numérica discreta / possivelmente ordinal"
        return "Quantitativa"

    if n_unicos == 2:
        return "Categórica binária"

    return "Categórica nominal/ordinal"


def tabela_descritiva_numerica(df):
    numericas = df.select_dtypes(include=np.number)

    if numericas.empty:
        return pd.DataFrame()

    linhas = []

    for coluna in numericas.columns:
        x = numericas[coluna].dropna()

        linhas.append({
            "Variável": coluna,
            "N": len(x),
            "Ausentes": df[coluna].isna().sum(),
            "Média": x.mean() if len(x) else np.nan,
            "DP": x.std(ddof=1) if len(x) > 1 else np.nan,
            "Erro padrão": x.sem(ddof=1) if len(x) > 1 else np.nan,
            "Mediana": x.median() if len(x) else np.nan,
            "Q1": x.quantile(0.25) if len(x) else np.nan,
            "Q3": x.quantile(0.75) if len(x) else np.nan,
            "Mínimo": x.min() if len(x) else np.nan,
            "Máximo": x.max() if len(x) else np.nan,
            "Amplitude": (x.max() - x.min()) if len(x) else np.nan,
            "CV (%)": (100 * x.std(ddof=1) / x.mean())
            if len(x) > 1 and x.mean() != 0 else np.nan,
            "Assimetria": x.skew() if len(x) > 2 else np.nan,
            "Curtose": x.kurt() if len(x) > 3 else np.nan,
        })

    return pd.DataFrame(linhas)


def mostrar_teste(nome, motivo, pressupostos=None, alternativa=None, observacao=None):
    st.success(f"### Sugestão: {nome}")
    st.markdown(f"**Por quê?** {motivo}")

    if pressupostos:
        st.markdown("**Verifique antes de usar:**")
        for item in pressupostos:
            st.markdown(f"- {item}")

    if alternativa:
        st.markdown(f"**Alternativa/observação:** {alternativa}")

    if observacao:
        st.warning(observacao)


# ============================================================
# ABAS
# ============================================================

aba_dados, aba_tutor, aba_mapa = st.tabs(
    ["1. Dados e estatística descritiva", "2. Tutor para escolha do teste", "3. Mapa dos testes"]
)

# ============================================================
# ABA 1 — DADOS
# ============================================================

with aba_dados:

    st.header("Importar arquivo CSV")

    arquivo = st.file_uploader(
        "Selecione um arquivo CSV com cabeçalho na primeira linha",
        type=["csv", "txt"]
    )

    if arquivo is not None:
        try:
            df = carregar_csv(arquivo)
            st.session_state["dados"] = df

            st.success(
                f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas."
            )

            st.subheader("Pré-visualização")
            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("Estrutura das variáveis")

            estrutura = pd.DataFrame({
                "Variável": df.columns,
                "Tipo no arquivo": [str(df[c].dtype) for c in df.columns],
                "Classificação sugerida": [classificar_coluna(df[c]) for c in df.columns],
                "N válido": [df[c].notna().sum() for c in df.columns],
                "Ausentes": [df[c].isna().sum() for c in df.columns],
                "Valores únicos": [df[c].nunique(dropna=True) for c in df.columns],
            })

            st.dataframe(estrutura, use_container_width=True)

            st.subheader("Estatística descritiva — variáveis quantitativas")

            desc = tabela_descritiva_numerica(df)

            if desc.empty:
                st.info("Nenhuma variável numérica foi identificada.")
            else:
                st.dataframe(
                    desc.style.format(precision=4, na_rep=""),
                    use_container_width=True
                )

            st.subheader("Frequências — variáveis categóricas")

            categorias = [
                c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                or df[c].nunique(dropna=True) <= 10
            ]

            if categorias:
                var_cat = st.selectbox(
                    "Escolha uma variável para visualizar frequências",
                    categorias
                )

                freq = (
                    df[var_cat]
                    .fillna("[Ausente]")
                    .value_counts(dropna=False)
                    .rename_axis("Categoria")
                    .reset_index(name="Frequência")
                )

                freq["Percentual (%)"] = 100 * freq["Frequência"] / len(df)

                st.dataframe(
                    freq.style.format({"Percentual (%)": "{:.2f}"}),
                    use_container_width=True
                )

                st.bar_chart(
                    freq.set_index("Categoria")["Frequência"]
                )

        except Exception as erro:
            st.error(f"Não foi possível ler o arquivo: {erro}")

    else:
        st.info(
            "Carregue um CSV para obter a descrição automática das colunas. "
            "O Tutor Estatístico também pode ser usado sem arquivo."
        )

# ============================================================
# ABA 2 — TUTOR DE ESCOLHA DO TESTE
# ============================================================

with aba_tutor:

    st.header("Tutor para escolha do teste estatístico")

    st.markdown(
        "Responda às perguntas com base no **objetivo científico e no desenho do estudo**. "
        "O programa fornecerá uma sugestão educativa."
    )

    objetivo = st.radio(
        "1. Qual é o principal objetivo da análise?",
        [
            "Comparar grupos ou condições",
            "Avaliar associação entre variáveis categóricas",
            "Avaliar correlação entre duas variáveis",
            "Estimar associação por Odds Ratio",
            "Modelar/predizer uma variável",
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # COMPARAÇÃO
    # --------------------------------------------------------

    if objetivo == "Comparar grupos ou condições":

        resposta = st.selectbox(
            "2. Qual é o tipo da variável que você deseja comparar?",
            [
                "Quantitativa contínua/discreta",
                "Ordinal",
                "Categórica binária",
                "Categórica nominal com mais de 2 categorias",
            ]
        )

        # QUANTITATIVA / ORDINAL
        if resposta in ["Quantitativa contínua/discreta", "Ordinal"]:

            n_condicoes = st.radio(
                "3. Quantos grupos/condições serão comparados?",
                ["2", "3 ou mais"]
            )

            dependencia = st.radio(
                "4. Como são as observações?",
                [
                    "Grupos independentes",
                    "Pareadas / mesmas pessoas / medidas repetidas",
                ]
            )

            if resposta == "Quantitativa contínua/discreta":
                parametrico = st.radio(
                    "5. É razoável utilizar um teste paramétrico?",
                    [
                        "Sim / aproximadamente",
                        "Não / distribuição muito assimétrica, ordinal ou com outliers importantes",
                        "Não sei",
                    ],
                    help=(
                        "A decisão não deve depender apenas de um teste de normalidade. "
                        "Considere distribuição, resíduos, outliers, tamanho amostral e desenho."
                    )
                )
            else:
                parametrico = "Não / distribuição muito assimétrica, ordinal ou com outliers importantes"

            if n_condicoes == "2":
                if dependencia == "Grupos independentes":
                    if parametrico == "Sim / aproximadamente":
                        mostrar_teste(
                            "Teste t para amostras independentes",
                            "Há dois grupos independentes e uma variável-resposta quantitativa.",
                            [
                                "Independência entre participantes/grupos",
                                "Distribuição/resíduos suficientemente adequados ao modelo",
                                "Avaliar homogeneidade das variâncias; se necessário, preferir a versão de Welch",
                            ],
                            "Quando as variâncias são diferentes, o teste t de Welch geralmente é preferível ao t clássico."
                        )
                    elif parametrico.startswith("Não /"):
                        mostrar_teste(
                            "Teste de Mann–Whitney",
                            "Há dois grupos independentes e a análise não seguirá a abordagem paramétrica.",
                            [
                                "Observações independentes",
                                "Variável ao menos ordinal",
                            ],
                            "Mann–Whitney não é simplesmente um 'teste de medianas'. A interpretação depende das distribuições dos grupos."
                        )
                    else:
                        st.info(
                            "Para decidir entre **teste t independente** e **Mann–Whitney**, "
                            "observe a distribuição, outliers, escala de medida, tamanho amostral e "
                            "o que exatamente deseja comparar."
                        )

                else:
                    if parametrico == "Sim / aproximadamente":
                        mostrar_teste(
                            "Teste t para amostras pareadas",
                            "Há duas medidas relacionadas, como antes/depois ou pares correspondentes.",
                            [
                                "As observações formam pares corretamente identificados",
                                "A distribuição relevante é a das diferenças entre os pares",
                            ],
                            "O pressuposto de normalidade, quando considerado, refere-se principalmente às diferenças."
                        )
                    elif parametrico.startswith("Não /"):
                        mostrar_teste(
                            "Teste de Wilcoxon para postos sinalizados",
                            "Há duas medidas relacionadas e a abordagem não paramétrica é mais apropriada.",
                            [
                                "Dados pareados",
                                "Variável ao menos ordinal",
                            ],
                            "O teste trabalha com a distribuição das diferenças e exige atenção especial a empates e diferenças iguais a zero."
                        )
                    else:
                        st.info(
                            "Para duas medidas relacionadas, as escolhas principais aqui são "
                            "**t pareado** ou **Wilcoxon**."
                        )

            else:
                if dependencia == "Grupos independentes":
                    if parametrico == "Sim / aproximadamente":
                        mostrar_teste(
                            "ANOVA de uma via",
                            "Há três ou mais grupos independentes e uma variável-resposta quantitativa.",
                            [
                                "Independência",
                                "Resíduos aproximadamente adequados ao modelo",
                                "Avaliar homogeneidade das variâncias",
                            ],
                            "Se a ANOVA global for significativa, uma análise pós-hoc apropriada é necessária."
                        )
                    elif parametrico.startswith("Não /"):
                        mostrar_teste(
                            "Teste de Kruskal–Wallis",
                            "Há três ou mais grupos independentes e será usada uma abordagem não paramétrica.",
                            [
                                "Observações independentes",
                                "Variável ao menos ordinal",
                            ],
                            "Se significativo, utilize comparações pós-hoc apropriadas com correção para múltiplos testes."
                        )
                    else:
                        st.info(
                            "Para três ou mais grupos independentes, considere **ANOVA de uma via** "
                            "ou **Kruskal–Wallis**, conforme o desenho e as características dos dados."
                        )

                else:
                    if parametrico == "Sim / aproximadamente":
                        mostrar_teste(
                            "ANOVA de uma via com medidas repetidas",
                            "Há três ou mais condições medidas nos mesmos participantes.",
                            [
                                "Medidas repetidas corretamente organizadas",
                                "Avaliar os resíduos",
                                "Avaliar esfericidade quando aplicável",
                            ],
                            "Se a esfericidade for violada, podem ser necessárias correções como Greenhouse–Geisser."
                        )
                    elif parametrico.startswith("Não /"):
                        mostrar_teste(
                            "Teste de Friedman",
                            "Há três ou mais medidas/condições relacionadas e será utilizada uma abordagem não paramétrica.",
                            [
                                "Mesmos indivíduos/blocos avaliados em todas as condições",
                                "Variável ao menos ordinal",
                            ],
                            "Se significativo, realize pós-hoc para localizar as diferenças."
                        )
                    else:
                        st.info(
                            "Para três ou mais medidas relacionadas, considere **ANOVA de medidas repetidas** "
                            "ou **Friedman**."
                        )

            st.divider()

            dois_fatores = st.checkbox(
                "Meu desenho possui duas variáveis independentes/fatores (por exemplo: grupo × sexo, tratamento × tempo)."
            )

            if dois_fatores:
                st.warning(
                    "**Considere ANOVA de duas vias** quando a variável-resposta é quantitativa "
                    "e há dois fatores categóricos. Ela permite testar os efeitos principais dos "
                    "dois fatores e, principalmente, a interação entre eles. "
                    "Se houver medidas repetidas ou desenho misto, é necessário especificar essa estrutura; "
                    "uma ANOVA de duas vias simples pode não ser adequada."
                )

        # CATEGÓRICA BINÁRIA
        elif resposta == "Categórica binária":

            n_condicoes = st.radio(
                "3. Quantas condições/grupos serão comparados?",
                ["2", "3 ou mais"]
            )

            dependencia = st.radio(
                "4. As observações são independentes ou relacionadas?",
                ["Independentes", "Pareadas / medidas repetidas"]
            )

            if dependencia == "Pareadas / medidas repetidas":
                if n_condicoes == "2":
                    mostrar_teste(
                        "Teste de McNemar",
                        "O desfecho é binário e há duas observações relacionadas por indivíduo.",
                        [
                            "Dados pareados",
                            "Desfecho dicotômico",
                        ],
                        "Exemplo clássico: resposta sim/não antes e depois de uma intervenção."
                    )
                else:
                    mostrar_teste(
                        "Teste Q de Cochran",
                        "O desfecho é binário e existem três ou mais condições relacionadas.",
                        [
                            "Mesmos indivíduos avaliados nas diferentes condições",
                            "Resposta dicotômica",
                        ],
                        "Pode ser entendido como uma extensão do teste de McNemar para mais de duas condições."
                    )

            else:
                mostrar_teste(
                    "Qui-quadrado de independência ou teste exato de Fisher",
                    "Há variáveis categóricas medidas em grupos independentes.",
                    [
                        "Observações independentes",
                        "Avaliar frequências esperadas nas células da tabela",
                    ],
                    "Quando frequências esperadas são pequenas, especialmente em tabelas 2×2, o teste exato de Fisher pode ser preferível."
                )

        else:
            dependencia = st.radio(
                "3. As observações são independentes?",
                ["Sim", "Não / medidas repetidas"]
            )

            if dependencia == "Sim":
                mostrar_teste(
                    "Qui-quadrado de independência",
                    "A comparação envolve variáveis categóricas em grupos independentes.",
                    [
                        "Observações independentes",
                        "Frequências esperadas suficientemente adequadas para a aproximação do qui-quadrado",
                    ],
                    "O teste G (razão de verossimilhanças) é uma alternativa para tabelas de contingência."
                )
            else:
                st.warning(
                    "Para respostas nominais com mais de duas categorias e medidas repetidas, "
                    "os testes listados no escopo inicial não cobrem adequadamente todas as situações. "
                    "Pode ser necessário usar modelos para dados categóricos ou métodos específicos."
                )

    # --------------------------------------------------------
    # ASSOCIAÇÃO CATEGÓRICA
    # --------------------------------------------------------

    elif objetivo == "Avaliar associação entre variáveis categóricas":

        tipo_problema = st.radio(
            "2. Qual é a situação?",
            [
                "Uma variável categórica: comparar frequências observadas com frequências esperadas",
                "Duas variáveis categóricas independentes",
                "Duas respostas binárias pareadas",
                "Resposta binária em 3 ou mais condições repetidas",
            ]
        )

        if tipo_problema.startswith("Uma variável"):
            mostrar_teste(
                "Qui-quadrado de aderência",
                "Você deseja verificar se as frequências observadas em uma única variável categórica seguem uma distribuição esperada.",
                [
                    "Categorias mutuamente exclusivas",
                    "Observações independentes",
                    "Frequências esperadas suficientemente grandes para a aproximação",
                ],
                "As proporções esperadas devem ser definidas por uma hipótese substantiva, não escolhidas após observar os dados."
            )

        elif tipo_problema == "Duas variáveis categóricas independentes":
            pequenas = st.radio(
                "3. Existem células com frequências esperadas pequenas?",
                ["Não / provavelmente não", "Sim / provavelmente sim", "Não sei"]
            )

            if pequenas == "Sim / provavelmente sim":
                mostrar_teste(
                    "Teste exato de Fisher",
                    "A análise envolve associação entre variáveis categóricas e há frequências esperadas pequenas.",
                    [
                        "Observações independentes",
                    ],
                    "É especialmente comum para tabelas 2×2."
                )
            elif pequenas == "Não / provavelmente não":
                escolha = st.radio(
                    "4. Qual abordagem deseja estudar?",
                    ["Qui-quadrado de independência", "Teste G (razão de verossimilhanças)"]
                )

                if escolha.startswith("Qui"):
                    mostrar_teste(
                        "Qui-quadrado de independência",
                        "Você deseja verificar associação entre duas variáveis categóricas em observações independentes.",
                        [
                            "Independência",
                            "Frequências esperadas adequadas",
                        ]
                    )
                else:
                    mostrar_teste(
                        "Teste G",
                        "O teste G utiliza a razão de verossimilhanças para comparar frequências observadas e esperadas.",
                        [
                            "Observações independentes",
                            "A aproximação assintótica deve ser adequada",
                        ],
                        "Em amostras grandes, costuma produzir conclusões semelhantes às do qui-quadrado de Pearson."
                    )
            else:
                st.info(
                    "Construa primeiro a tabela de contingência e examine as **frequências esperadas**. "
                    "Isso ajuda a decidir entre qui-quadrado e métodos exatos."
                )

        elif tipo_problema == "Duas respostas binárias pareadas":
            mostrar_teste(
                "Teste de McNemar",
                "São duas medidas binárias relacionadas.",
                [
                    "Pareamento correto",
                    "Resposta binária",
                ]
            )

        else:
            mostrar_teste(
                "Teste Q de Cochran",
                "Há uma resposta binária medida em três ou mais condições relacionadas.",
                [
                    "Mesmos indivíduos nas condições",
                    "Resposta binária",
                ]
            )

    # --------------------------------------------------------
    # CORRELAÇÃO
    # --------------------------------------------------------

    elif objetivo == "Avaliar correlação entre duas variáveis":

        tipo_vars = st.radio(
            "2. Como são as duas variáveis?",
            [
                "Duas variáveis quantitativas",
                "Pelo menos uma é ordinal ou a relação é monotônica, mas não necessariamente linear",
            ]
        )

        if tipo_vars == "Duas variáveis quantitativas":
            linear = st.radio(
                "3. A relação esperada é aproximadamente linear e sem outliers muito influentes?",
                ["Sim", "Não / não sei"]
            )

            if linear == "Sim":
                mostrar_teste(
                    "Correlação linear de Pearson",
                    "Há duas variáveis quantitativas e o interesse é medir a intensidade de uma relação linear.",
                    [
                        "Relação aproximadamente linear",
                        "Avaliar outliers influentes",
                        "Para inferência clássica, verificar os pressupostos associados ao modelo",
                    ],
                    "Correlação não implica causalidade."
                )
            else:
                mostrar_teste(
                    "Correlação de Spearman",
                    "A correlação de postos é apropriada para relações monotônicas, dados ordinais ou situações em que Pearson não é uma boa descrição.",
                    [
                        "Relação aproximadamente monotônica para uma interpretação simples",
                    ],
                    "Spearman detecta associação monotônica, não apenas linear."
                )
        else:
            mostrar_teste(
                "Correlação de Spearman",
                "A análise envolve escalas ordinais ou uma relação monotônica que não precisa ser linear.",
                [
                    "Pares de observações correspondentes",
                ]
            )

    # --------------------------------------------------------
    # ODDS RATIO
    # --------------------------------------------------------

    elif objetivo == "Estimar associação por Odds Ratio":

        st.markdown(
            "O **Odds Ratio (OR)** é especialmente natural quando exposição e desfecho são binários "
            "e podem ser organizados em uma tabela 2×2."
        )

        mostrar_teste(
            "Odds Ratio",
            "Você deseja quantificar a magnitude da associação entre uma exposição e um desfecho binário.",
            [
                "Definir claramente a categoria de referência",
                "Verificar se o desenho do estudo permite a interpretação pretendida",
                "Apresentar intervalo de confiança, normalmente de 95%",
            ],
            "OR = 1 indica ausência de associação na escala de odds; OR > 1 e OR < 1 dependem da codificação das categorias.",
            "Odds Ratio não é, em geral, igual a risco relativo. Quando o desfecho é comum, as duas medidas podem diferir consideravelmente."
        )

    # --------------------------------------------------------
    # MODELAGEM
    # --------------------------------------------------------

    elif objetivo == "Modelar/predizer uma variável":

        resposta_modelo = st.selectbox(
            "2. Qual é o tipo da variável dependente (Y)?",
            [
                "Quantitativa",
                "Binária",
                "Outra",
            ]
        )

        if resposta_modelo == "Quantitativa":

            forma = st.radio(
                "3. Como espera que Y se relacione com o(s) preditor(es)?",
                [
                    "Relação aproximadamente linear nos parâmetros",
                    "Relação claramente não linear com uma função teórica/biológica definida",
                    "Ainda não sei",
                ]
            )

            if forma == "Relação aproximadamente linear nos parâmetros":
                mostrar_teste(
                    "Regressão linear",
                    "A variável dependente é quantitativa e a relação pode ser representada por um modelo linear nos parâmetros.",
                    [
                        "Linearidade da estrutura média",
                        "Independência dos erros",
                        "Homoscedasticidade",
                        "Avaliar resíduos e observações influentes",
                        "Normalidade dos resíduos é relevante para certas formas de inferência em amostras pequenas",
                    ],
                    "O modelo pode incluir uma ou várias variáveis preditoras."
                )

            elif forma.startswith("Relação claramente"):
                mostrar_teste(
                    "Regressão não linear",
                    "Existe uma forma funcional não linear com significado científico ou empírico.",
                    [
                        "Definir a função/modelo antes do ajuste",
                        "Escolher valores iniciais adequados para os parâmetros",
                        "Avaliar resíduos, identificabilidade e qualidade do ajuste",
                    ],
                    "Exemplos incluem crescimento exponencial, modelos logísticos, curvas de saturação e outras funções específicas.",
                    "Não escolha regressão não linear apenas porque o diagrama de dispersão 'parece curvo'. Primeiro defina qual mecanismo ou função é plausível."
                )

            else:
                st.info(
                    "Comece pelo diagrama de dispersão e por uma hipótese sobre a forma da relação. "
                    "Se o modelo for linear nos parâmetros, use regressão linear; se houver uma função "
                    "não linear substantivamente definida, considere regressão não linear."
                )

        elif resposta_modelo == "Binária":
            st.warning(
                "Para uma variável dependente binária, o método de regressão mais típico seria "
                "**regressão logística**, que ainda não estava na sua lista inicial. "
                "Vale muito a pena incluí-la em uma próxima versão do Tutor."
            )

        else:
            st.warning(
                "O tipo de modelo depende da distribuição e da escala da variável-resposta. "
                "Modelos lineares simples não cobrem adequadamente todos os tipos de desfecho."
            )

# ============================================================
# ABA 3 — MAPA DIDÁTICO DOS TESTES
# ============================================================

with aba_mapa:

    st.header("Mapa resumido dos testes incluídos")

    mapa = pd.DataFrame([
        ["Teste t independente", "Comparação", "Quantitativa", "2 grupos", "Independentes", "Paramétrico"],
        ["Teste t pareado", "Comparação", "Quantitativa", "2 condições", "Pareadas", "Paramétrico"],
        ["Mann–Whitney", "Comparação", "Ordinal/quantitativa", "2 grupos", "Independentes", "Não paramétrico"],
        ["Wilcoxon", "Comparação", "Ordinal/quantitativa", "2 condições", "Pareadas", "Não paramétrico"],
        ["Qui-quadrado de aderência", "Frequências", "Categórica", "1 variável", "Independentes", "Categórico"],
        ["Qui-quadrado de independência", "Associação", "Categóricas", "Tabela de contingência", "Independentes", "Categórico"],
        ["Fisher", "Associação", "Categóricas", "Geralmente 2×2", "Independentes", "Exato"],
        ["Teste G", "Associação", "Categóricas", "Tabela de contingência", "Independentes", "Razão de verossimilhanças"],
        ["McNemar", "Comparação/associação", "Binária", "2 condições", "Pareadas", "Categórico"],
        ["Q de Cochran", "Comparação", "Binária", "3+ condições", "Repetidas", "Categórico"],
        ["ANOVA de uma via", "Comparação", "Quantitativa", "3+ grupos", "Independentes", "Paramétrico"],
        ["ANOVA 1 via medidas repetidas", "Comparação", "Quantitativa", "3+ condições", "Repetidas", "Paramétrico"],
        ["ANOVA de duas vias", "Comparação/modelagem", "Quantitativa", "2 fatores", "Depende do desenho", "Paramétrico"],
        ["Kruskal–Wallis", "Comparação", "Ordinal/quantitativa", "3+ grupos", "Independentes", "Não paramétrico"],
        ["Friedman", "Comparação", "Ordinal/quantitativa", "3+ condições", "Repetidas", "Não paramétrico"],
        ["Pearson", "Correlação", "2 quantitativas", "-", "Pares", "Linear"],
        ["Spearman", "Correlação", "Ordinal/quantitativa", "-", "Pares", "Por postos"],
        ["Odds Ratio", "Associação", "Binárias", "Tabela 2×2", "Depende do desenho", "Medida de efeito"],
        ["Regressão linear", "Modelagem", "Y quantitativo", "1+ preditores", "Depende do desenho", "Modelo linear"],
        ["Regressão não linear", "Modelagem", "Y quantitativo", "1+ preditores", "Depende do desenho", "Modelo não linear"],
    ], columns=[
        "Teste/Método",
        "Objetivo",
        "Tipo de variável",
        "Grupos/condições",
        "Relação entre observações",
        "Família"
    ])

    st.dataframe(mapa, use_container_width=True, hide_index=True)

    st.subheader("Uma forma simples de pensar")

    st.markdown(
        """
        **Comparação**
        - 2 grupos independentes → t independente ou Mann–Whitney
        - 2 medidas relacionadas → t pareado ou Wilcoxon
        - 3+ grupos independentes → ANOVA de uma via ou Kruskal–Wallis
        - 3+ medidas relacionadas → ANOVA de medidas repetidas ou Friedman
        - 2 fatores → ANOVA de duas vias

        **Dados categóricos**
        - Uma distribuição observada versus esperada → qui-quadrado de aderência
        - Duas variáveis categóricas independentes → qui-quadrado / G / Fisher
        - Duas respostas binárias pareadas → McNemar
        - 3+ respostas binárias repetidas → Q de Cochran

        **Relação entre variáveis**
        - Correlação linear entre quantitativas → Pearson
        - Associação monotônica/ordinal → Spearman
        - Associação entre duas variáveis binárias → Odds Ratio
        - Predição de Y quantitativo → regressão linear ou não linear
        """
    )

    st.warning(
        "A árvore de decisão é uma ferramenta didática. Em estudos reais, desenho experimental, "
        "estrutura de dependência, covariáveis, dados ausentes, distribuição dos resíduos e objetivo "
        "inferencial podem indicar modelos mais avançados."
    )
