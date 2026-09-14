# InsurGuard AI

**InsurGuard AI** es un prototipo de TFM para el preanálisis de siniestros de seguros de automóvil. El sistema combina un canal de entrada mediante **Telegram**, detección de fraude con Machine Learning, una consola web para el **supervisor/especialista**, consulta de pólizas mediante **RAG con Sentence Transformers + PostgreSQL + pgvector**, explicabilidad y monitorización MLOps.

> El sistema actúa como herramienta de apoyo al supervisor. La predicción de fraude y las respuestas sobre pólizas no sustituyen la decisión humana final.

## Accesos rápidos

- **Bot de Telegram:** https://t.me/insurguard_ai_bot
- **Consola del supervisor:** http://127.0.0.1:8000/admin
- **Swagger / API:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health
- **Estado Telegram:** http://127.0.0.1:8000/telegram/info
- **Estado RAG:** http://127.0.0.1:8000/rag/info

---

## Arquitectura actual

```text
Cliente
  │
  ▼
Telegram Bot
  │
  ▼
Telegram Intake Agent
  │
  ├── captura guiada del siniestro
  ├── botones inline
  └── adjuntos opcionales como evidencia
  │
  ▼
Validación del claim
  │
  ▼
Fraud Detection Agent
  │
  ├── Logistic Regression L1
  ├── fraud_score
  ├── nivel de riesgo
  └── explicación local SHAP
  │
  ▼
MLOps

Supervisor /admin
  │
  ├── Bandeja de expedientes
  ├── filtro por estado
  ├── apertura de documentos del cliente
  ├── análisis de fraude + SHAP
  ├── ground truth
  └── asistente RAG de pólizas
           │
           ├── Sentence Transformers
           ├── PostgreSQL + pgvector
           ├── fragmento exacto recuperado
           └── apertura de la póliza en la página fuente
```

El **cliente de Telegram no consulta pólizas ni coberturas**. Esa funcionalidad está reservada al supervisor desde `/admin`.

Los documentos adjuntados por el cliente se almacenan para que el supervisor pueda abrirlos y revisarlos. El **OCR se utiliza únicamente durante la indexación de pólizas**, página a página, cuando una página del PDF no contiene texto nativo suficiente.

---

# 1. Requisitos

La forma recomendada de ejecutar el proyecto es con Docker.

Necesitas:

- **Docker Desktop** con Docker Compose.
- Git, si vas a clonar el repositorio.
- Un token válido del bot de Telegram `@insurguard_ai_bot`.
- Las pólizas PDF indicadas en `data/policies/` si quieres utilizar el RAG.

No necesitas instalar PostgreSQL, pgvector, Tesseract ni las dependencias Python manualmente si utilizas Docker. La imagen de la API las instala automáticamente.

---

# 2. Preparar el proyecto

Clona el repositorio o descomprime el proyecto y entra en su carpeta:

```powershell
cd C:\ruta\a\insuguardAI
```

La estructura principal debe contener al menos:

```text
insuguardAI/
├── backend/
├── data/
│   ├── kaggle/
│   │   └── fraud_oracle.csv
│   ├── policies/
│   │   ├── manifest.json
│   │   └── README.md
│   └── claims/
├── models/
│   ├── fraud_pipeline.joblib
│   └── model_metadata.json
├── scripts/
├── tests/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 3. Añadir las pólizas del RAG

Los PDF de las pólizas pueden mantenerse fuera de GitHub. Antes de levantar el sistema completo, copia los documentos en:

```text
data/policies/
```

El `manifest.json` actual espera estos nombres:

```text
data/policies/
├── aviva_short_term_car_policy.pdf
├── axa_car_plus_policy.pdf
├── axa_car_policy.pdf
├── manifest.json
└── README.md
```

Correspondencia:

| `policy_id` | Archivo |
|---|---|
| `AVIVA-SHORT-TERM` | `aviva_short_term_car_policy.pdf` |
| `AXA-CAR-PLUS` | `axa_car_plus_policy.pdf` |
| `AXA-CAR` | `axa_car_policy.pdf` |

**Importante:** el arranque Docker ejecuta automáticamente `scripts/index_policies.py`. Si `manifest.json` referencia un PDF que no existe, la API no podrá completar la indexación. Por tanto, coloca primero los PDF o adapta el manifest a los documentos disponibles.

Para no subir los PDF al repositorio puedes añadir a `.gitignore`:

```gitignore
# Insurance policy PDFs
data/policies/*.pdf

!data/policies/README.md
!data/policies/manifest.json
!data/policies/.gitkeep
```

Durante la indexación, InsurGuard intenta extraer texto de cada página de la póliza. Si una página no tiene texto nativo suficiente, utiliza Tesseract OCR únicamente para esa página. Después genera embeddings y almacena los chunks en PostgreSQL + pgvector.

---

# 4. Configurar Telegram

Copia el archivo de ejemplo:

```powershell
Copy-Item .env.example .env
```

En `.env` configura el token del bot:

```env
TELEGRAM_BOT_TOKEN=TU_TOKEN_DE_BOTFATHER
TELEGRAM_POLLING_ENABLED=true
TELEGRAM_POLL_TIMEOUT=25
TELEGRAM_DROP_PENDING_UPDATES=false
```

No subas `.env` a GitHub.

El bot del proyecto es:

**https://t.me/insurguard_ai_bot**

Si necesitas regenerar o sustituir su token, hazlo desde `@BotFather` y actualiza `TELEGRAM_BOT_TOKEN`.

---

# 5. Levantar todo con Docker

Desde la raíz del proyecto:

```powershell
docker compose up --build -d
```

Esto levanta:

```text
insurguard-postgres    PostgreSQL 16 + pgvector
insurguard-api         FastAPI + Telegram + ML + RAG + MLOps + supervisor
```

Durante el arranque de la API se ejecuta también la sincronización de las pólizas con pgvector.

Puedes comprobar los contenedores con:

```powershell
docker compose ps
```

Y seguir los logs de la API con:

```powershell
docker compose logs -f insurguard-api
```

Cuando aparezca que Uvicorn está ejecutándose, abre:

```text
http://127.0.0.1:8000/health
```

Debe responder correctamente antes de continuar.

---

# 6. Comprobar que Telegram está conectado

Abre:

```text
http://127.0.0.1:8000/telegram/info
```

Con el token correcto debes ver una configuración equivalente a:

```text
configured: true
polling_enabled: true
live_enabled: true
mode: telegram_long_polling
poller.running: true
```

También puedes probar el token desde el contenedor:

```powershell
docker compose exec insurguard-api python scripts/check_telegram.py
```

El proyecto utiliza **long polling (`getUpdates`)**, por lo que no necesita ngrok, webhook público ni dominio propio.

---

# 7. Probar un claim desde Telegram

Abre:

**https://t.me/insurguard_ai_bot**

Pulsa **Start** o envía:

```text
/start
```

Después puedes iniciar un expediente mediante:

```text
NUEVO
```

También están disponibles:

```text
/nuevo
/estado
/analizar
/cancelar
/ayuda
```

El bot realiza una captura guiada del siniestro con explicaciones y botones inline. A partir de las respuestas genera las **30 variables de entrada** requeridas por el modelo de fraude.

Los claims creados mediante Telegram utilizan identificadores del tipo:

```text
TG-...
```

### Adjuntar documentación

El cliente puede adjuntar PDFs o imágenes de forma opcional. Se guardan en:

```text
data/claims/TG-.../documents/
```

Estos archivos se conservan como **evidencia original** para que el supervisor pueda abrirlos desde `/admin`.

No se utilizan para generar la predicción del modelo ni se procesan mediante OCR.

Cuando hayas terminado la captura, ejecuta:

```text
ANALIZAR
```

El sistema ejecutará la validación y el modelo de fraude y registrará la predicción para la monitorización MLOps.

---

# 8. Consola del supervisor

Abre:

```text
http://127.0.0.1:8000/admin
```

La consola está orientada al tramitador/supervisor y permite revisar el expediente completo.

## Bandeja de expedientes

La bandeja permite buscar y filtrar por:

```text
Todos los estados
Pendiente de análisis
Ground truth pendiente
Ground truth confirmado
Datos de referencia
```

Al abrir un expediente se muestran, entre otros:

- `fraud_score`;
- umbral de decisión;
- nivel de riesgo;
- necesidad de revisión manual;
- factores explicativos SHAP;
- datos estructurados del claim;
- documentos originales aportados por el cliente;
- traza técnica de ejecución;
- estado de monitorización;
- ground truth, cuando exista.

### Abrir documentos del cliente

Los adjuntos del claim pueden abrirse directamente desde el expediente para que el supervisor contraste la información declarada con la evidencia original.

---

# 9. Ground truth del supervisor

Para claims externos analizados, el supervisor puede registrar posteriormente el resultado real:

```text
✓ Confirmar NO fraude
⚠ Confirmar FRAUDE
```

Internamente utiliza:

```text
PUT /mlops/production-claims/{claim_id}/ground-truth
```

El ground truth:

- no modifica la predicción histórica;
- permite comparar predicción y resultado real;
- alimenta las métricas de rendimiento de producción;
- sirve como señal para valorar degradación y posible reentrenamiento futuro.

Los claims de demostración procedentes de Kaggle muestran su etiqueta de referencia, pero esta no es editable desde el supervisor.

---

# 10. Asistente RAG de pólizas

Las consultas sobre pólizas son una funcionalidad exclusiva del **supervisor**.

Desde `/admin` puedes seleccionar una póliza y realizar preguntas libres como:

```text
¿Qué exclusiones pueden afectar a este siniestro?

¿Qué franquicia aplica a los daños propios?

¿Qué documentación exige la póliza?

¿Qué límites o condiciones deben comprobarse?
```

El flujo es:

```text
Pregunta del supervisor
        ↓
Sentence Transformers
        ↓
Embedding de consulta
        ↓
PostgreSQL + pgvector
        ↓
Filtro estricto por policy_id
        ↓
Chunks más relevantes
        ↓
Evidencia para el supervisor
```

La interfaz muestra el fragmento exacto recuperado, score de similitud, identificador del chunk y páginas detectadas. El supervisor también puede abrir el PDF original de la póliza para contrastar la fuente.

La consulta RAG es independiente de la predicción de fraude y no crea una nueva predicción ni un evento MLOps.

---

# 11. Reindexar pólizas

Si añades, sustituyes o modificas los PDF de `data/policies/`, ejecuta:

```powershell
docker compose exec insurguard-api python scripts/index_policies.py --force
```

También puedes hacerlo desde Swagger:

```text
POST /rag/reindex?force=true
```

Consulta después:

```text
http://127.0.0.1:8000/rag/info
```

El estado de indexación permite comprobar los documentos y páginas procesadas.

---

# 12. Modelo de detección de fraude

El modelo de producción del prototipo es una **Regresión Logística L1 cost-sensitive**.

Configuración principal:

```text
solver = liblinear
penalty = l1
C = 0.001
class_weight = {0: 1, 1: 15.715447154471544}
threshold = 0.5748224375777465
```

Benchmark reproducido sobre el holdout independiente:

```text
PR-AUC:     0.526455
ROC-AUC:    0.755227
Recall:     0.9243
Precision:  0.1256
F1:         0.2211

Confusion matrix:
[[1708, 1191],
 [14,   171]]
```

El modelo usa las 30 variables explicativas del dataset `fraud_oracle.csv`. `FraudFound_P` se utiliza exclusivamente como target y nunca se introduce como feature durante inferencia.

Puedes verificar el benchmark con:

```powershell
docker compose exec insurguard-api python scripts/verify_benchmark.py
```

---

# 13. MLOps

La monitorización se almacena en:

```text
artifacts/monitoring/monitoring.db
```

Endpoints principales:

```text
GET /mlops/model-info
GET /mlops/model-metrics
GET /mlops/data-quality
GET /mlops/production-data-quality
GET /mlops/monitoring-summary
GET /mlops/drift-report
GET /mlops/prediction-drift
GET /mlops/production-performance
GET /mlops/retraining-status
PUT /mlops/production-claims/{claim_id}/ground-truth
```

El drift formal se calcula cuando existe volumen suficiente de claims externos. El reentrenamiento nunca se ejecuta automáticamente: el sistema únicamente genera una recomendación para revisión.

---

# 14. Comprobar el proyecto completo

Ejecuta los tests dentro del contenedor:

```powershell
docker compose exec insurguard-api pytest -q
```

Comprueba el benchmark:

```powershell
docker compose exec insurguard-api python scripts/verify_benchmark.py
```

Comprueba Telegram:

```powershell
docker compose exec insurguard-api python scripts/check_telegram.py
```

Swagger permite probar manualmente todos los endpoints:

```text
http://127.0.0.1:8000/docs
```

---

# 15. Comandos Docker útiles

### Ver estado

```powershell
docker compose ps
```

### Ver logs de la API

```powershell
docker compose logs -f insurguard-api
```

### Reiniciar únicamente la API

```powershell
docker compose restart insurguard-api
```

### Recargar un `.env` modificado sin reconstruir la imagen

```powershell
docker compose up -d --no-deps --force-recreate insurguard-api
```

### Reconstruir únicamente la API después de cambiar código

```powershell
docker compose up -d --no-deps --build --force-recreate insurguard-api
```

### Parar el sistema

```powershell
docker compose down
```

Esto conserva el volumen de PostgreSQL.

### Eliminar también la base PostgreSQL persistente

```powershell
docker compose down -v
```

**Atención:** `-v` elimina el volumen de PostgreSQL y, por tanto, el índice pgvector. Después habrá que indexar las pólizas de nuevo.

---

# 16. Flujo recomendado para una demostración

1. Ejecuta:

```powershell
docker compose up --build -d
```

2. Comprueba:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/telegram/info
http://127.0.0.1:8000/rag/info
```

3. Abre el bot:

**https://t.me/insurguard_ai_bot**

4. Envía:

```text
NUEVO
```

5. Completa el claim y, si quieres, adjunta documentación.

6. Ejecuta:

```text
ANALIZAR
```

7. Abre la consola:

```text
http://127.0.0.1:8000/admin
```

8. Localiza el claim `TG-...` en la **Bandeja de expedientes**.

9. Revisa el `fraud_score`, riesgo, SHAP y documentos originales.

10. Desde el asistente de pólizas, selecciona una póliza y realiza una pregunta. Comprueba el fragmento exacto recuperado y abre el PDF fuente.

11. Cuando quieras simular el cierre del expediente, registra el ground truth desde el panel del supervisor.

12. Revisa la sección MLOps para comprobar que el claim externo y su resultado están monitorizados.

---

# 17. Solución de problemas

### Telegram aparece como no configurado

Comprueba `.env`:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_POLLING_ENABLED=true
```

Después recrea solamente la API:

```powershell
docker compose up -d --no-deps --force-recreate insurguard-api
```

Y revisa:

```text
http://127.0.0.1:8000/telegram/info
```

### La API no arranca después de quitar los PDF de GitHub

Comprueba que todos los ficheros referenciados en:

```text
data/policies/manifest.json
```

existan realmente dentro de:

```text
data/policies/
```

El indexador se ejecuta al iniciar el contenedor y detendrá el arranque si el manifest apunta a un PDF inexistente.

### He cambiado una póliza pero el RAG sigue mostrando el índice anterior

Fuerza la reindexación:

```powershell
docker compose exec insurguard-api python scripts/index_policies.py --force
```

### He cambiado solo `.env`

No reconstruyas toda la imagen:

```powershell
docker compose up -d --no-deps --force-recreate insurguard-api
```

### He cambiado código Python, HTML, CSS o JavaScript

Reconstruye únicamente la API:

```powershell
docker compose up -d --no-deps --build --force-recreate insurguard-api
```

---

# Tecnologías

- Python 3.11
- FastAPI
- Telegram Bot API
- scikit-learn
- SHAP
- PostgreSQL 16
- pgvector
- Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- PyMuPDF / pypdf
- Tesseract OCR para páginas de póliza que lo requieran
- Jinja2 + JavaScript para la consola del supervisor
- Docker / Docker Compose
- Pytest

---

## Nota académica

InsurGuard AI es un prototipo desarrollado con fines académicos. El modelo de fraude y el asistente de pólizas sirven como apoyo al análisis y no deben interpretarse como una decisión automática definitiva sobre fraude, cobertura o aceptación/rechazo de un siniestro.
