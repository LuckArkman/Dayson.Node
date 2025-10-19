#!/bin/bash
# ====================================================================
# SCRIPT DE TESTE DE ESTABILIDADE E MONITORAMENTO DE MEMÓRIA
#
# Este script executa o ciclo completo de treinamento da aplicação
# Galileu.Node, garantindo que o ambiente da GPU seja carregado
# corretamente e monitorando o uso de RAM e CPU.
#
# Uso: ./run_training_monitor.sh
# ====================================================================

# >>> INÍCIO DA CORREÇÃO PARA DETECÇÃO DA GPU <<<
# Força o carregamento do perfil do usuário para garantir que as variáveis de ambiente
# (especialmente LD_LIBRARY_PATH e outras necessárias para o OpenCL) estejam definidas.
# Scripts não-interativos muitas vezes não carregam esses perfis por padrão.
echo "Carregando perfil do usuário para garantir o ambiente da GPU..."
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
    echo "Fonte: ~/.bashrc"
elif [ -f "$HOME/.profile" ]; then
    source "$HOME/.profile"
    echo "Fonte: ~/.profile"
else
    echo "Aviso: Nenhum arquivo de perfil (~/.bashrc ou ~/.profile) encontrado."
fi
echo "-----------------------------------------------------"
# >>> FIM DA CORREÇÃO <<<

set -e

# Definição de Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Limpa a tela para uma nova execução
clear
echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}   TESTE DE ESTABILIDADE E MONITORAMENTO DE MEMÓRIA   ${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

# ========================================
# FASE 0: Limpeza do Ambiente
# ========================================
echo -e "${YELLOW}[FASE 0] Limpando ambiente de teste...${NC}"
rm -f build.log train_run.log memory_trace.csv
rm -f Dayson/Dayson.json Dayson/memory.bin Dayson/vocab.txt Dayson/lstm_cache_disk_only.bin 2>/dev/null || true
dotnet clean > /dev/null 2>&1
echo -e "${GREEN}✅ Ambiente limpo.${NC}"
echo ""

# ========================================
# FASE 1: Compilação
# ========================================
echo -e "${BLUE}[FASE 1] Compilando projeto...${NC}"
if dotnet build > build.log 2>&1; then
    echo -e "${GREEN}✅ Compilação bem-sucedida.${NC}"
else
    echo -e "${RED}❌ Erro de compilação. Verifique o arquivo 'build.log'.${NC}"
    tail -20 build.log
    exit 1
fi
echo ""

# ========================================
# FASE 2: Execução e Monitoramento
# ========================================
echo -e "${BLUE}[FASE 2] Iniciando treinamento e monitoramento...${NC}"
echo "O teste será executado por até 30 minutos ou até o processo terminar."
echo "As métricas de RAM e CPU serão coletadas a cada 5 segundos."
echo ""

TIMEOUT_SECONDS=1800 

timeout $TIMEOUT_SECONDS dotnet run > train_run.log 2>&1 &
PID=$!

trap 'echo -e "\n${YELLOW}🧹 Teste interrompido. Encerrando processo...${NC}"; kill $PID 2>/dev/null || true; exit 1' INT TERM

declare -a RAM_READINGS
declare -a CPU_READINGS
MONITOR_INTERVAL_SECONDS=5
i=0
STUCK_COUNT=0
LAST_RAM_MB=0

echo "Timestamp,Leitura,RAM_MB,CPU%" > memory_trace.csv

while kill -0 $PID 2>/dev/null; do
    sleep $MONITOR_INTERVAL_SECONDS

    RAM_KB=$(ps -p $PID -o rss= 2>/dev/null || echo 0)
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null || echo 0)
    RAM_MB=$((RAM_KB / 1024))

    RAM_READINGS+=($RAM_MB)
    CPU_READINGS+=($CPU)
    i=$((i+1))

    echo "$(date +%T),$i,$RAM_MB,$CPU" >> memory_trace.csv
    echo -ne "\r${YELLOW}Monitorando...${NC} Leitura $i: RAM = ${RAM_MB}MB | CPU = ${CPU}%   "

    if (( i > 10 )) && (( RAM_MB == LAST_RAM_MB )); then
        STUCK_COUNT=$((STUCK_COUNT+1))
    else
        STUCK_COUNT=0
    fi

    if (( STUCK_COUNT > 20 )); then
        echo -e "\n${RED}❌ Travamento detectado (RAM estável por muito tempo). Encerrando teste.${NC}"
        kill $PID 2>/dev/null || true
        break
    fi

    LAST_RAM_MB=$RAM_MB
done

wait $PID || true
echo -e "\n\n${GREEN}✅ Processo de treinamento concluído.${NC}"
echo ""

# ========================================
# FASE 3: Análise dos Resultados
# ========================================
echo -e "${BLUE}[FASE 3] Analisando resultados do monitoramento...${NC}"
echo ""

if [ ${#RAM_READINGS[@]} -lt 10 ]; then
    echo -e "${RED}❌ Dados de monitoramento insuficientes (${#RAM_READINGS[@]} leituras).${NC}"
    echo "O processo provavelmente falhou muito rápido. Verifique o log 'train_run.log' para detalhes:"
    tail -30 train_run.log
    exit 1
fi

RAM_MIN=$(printf '%s\n' "${RAM_READINGS[@]}" | sort -n | head -1)
RAM_MAX=$(printf '%s\n' "${RAM_READINGS[@]}" | sort -n | tail -1)
RAM_AVG=$(awk '{ total += $1; count++ } END { if (count > 0) print total/count; else print 0 }' <<< "${RAM_READINGS[*]}")
RAM_AVG=$(printf "%.0f" $RAM_AVG)
RAM_RANGE=$((RAM_MAX - RAM_MIN))
GROWTH=$((RAM_READINGS[-1] - RAM_READINGS[0]))

echo "📊 Estatísticas de Uso de RAM:"
echo "   • Duração do Monitoramento: $((i * MONITOR_INTERVAL_SECONDS)) segundos"
echo "   • Leituras Coletadas:      $i"
echo "   • Mínimo:                  ${RAM_MIN} MB"
echo "   • Máximo:                  ${RAM_MAX} MB"
echo "   • Média:                   ${RAM_AVG} MB"
echo "   • Variação (Pico - Mínimo):  ${RAM_RANGE} MB"
echo "   • Crescimento (Fim - Início): ${GROWTH} MB"
echo ""

# ========================================
# FASE 4: Veredito Final
# ========================================
echo -e "${BLUE}[FASE 4] Veredito de Estabilidade...${NC}"
echo ""

RAM_RANGE_THRESHOLD=800
GROWTH_THRESHOLD=400
RAM_MAX_THRESHOLD=16000

SCORE=100

if [ $RAM_RANGE -gt $RAM_RANGE_THRESHOLD ]; then
    echo -e "${RED}❌ Variação de RAM alta:${NC} ${RAM_RANGE}MB (limite: ${RAM_RANGE_THRESHOLD}MB). Isso pode indicar picos de alocação excessivos."
    SCORE=$((SCORE-30))
else
    echo -e "${GREEN}✅ Variação de RAM aceitável:${NC} ${RAM_RANGE}MB (limite: ${RAM_RANGE_THRESHOLD}MB)."
fi

if [ $GROWTH -gt $GROWTH_THRESHOLD ]; then
    echo -e "${RED}❌ Crescimento linear alto:${NC} ${GROWTH}MB (limite: ${GROWTH_THRESHOLD}MB). Forte indício de vazamento de memória."
    SCORE=$((SCORE-50))
else
    echo -e "${GREEN}✅ Crescimento linear baixo:${NC} ${GROWTH}MB (limite: ${GROWTH_THRESHOLD}MB). Sem sinal de vazamento contínuo."
fi

if [ $RAM_MAX -gt $RAM_MAX_THRESHOLD ]; then
    echo -e "${RED}❌ Pico de RAM excedeu o limite:${NC} ${RAM_MAX}MB (limite: ${RAM_MAX_THRESHOLD}MB)."
    SCORE=$((SCORE-20))
else
    echo -e "${GREEN}✅ Pico de RAM dentro do limite:${NC} ${RAM_MAX}MB (limite: ${RAM_MAX_THRESHOLD}MB)."
fi

echo ""
if [ $SCORE -ge 80 ]; then
    echo -e "${GREEN}✅ VEREDITO: SISTEMA ESTÁVEL (${SCORE}/100)${NC}"
elif [ $SCORE -ge 60 ]; then
    echo -e "${YELLOW}⚠️  VEREDITO: ESTABILIDADE PARCIAL (${SCORE}/100)${NC}"
else
    echo -e "${RED}❌ VEREDITO: INSTABILIDADE DETECTADA (${SCORE}/100)${NC}"
fi

echo ""
echo "-----------------------------------------------------"
echo "Logs e artefatos gerados:"
echo "  • build.log        (log de compilação)"
echo "  • train_run.log    (saída do console da aplicação)"
echo "  • memory_trace.csv (dados brutos de RAM/CPU para análise em planilhas)"
echo "-----------------------------------------------------"
echo ""
echo -e "${BLUE}Teste de monitoramento concluído.${NC}"