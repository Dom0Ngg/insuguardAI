# Pólizas utilizadas por InsurGuard AI

Esta carpeta contiene las pólizas utilizadas por el módulo RAG de InsurGuard AI.

Los documentos PDF originales no se incluyen en el repositorio por motivos de
tamaño, distribución y trazabilidad de las fuentes.

Las pólizas utilizadas durante el desarrollo son documentos públicos del ramo
de automóvil.

## Estructura esperada

Para ejecutar el módulo RAG deben colocarse manualmente los siguientes archivos
en esta carpeta:

- `AXA-CAR.pdf`
- `AXA-CAR-PLUS.pdf`
- `AVIVA-SHORT-TERM.pdf`

La estructura debe quedar:

data/policies/
├── AXA-CAR.pdf
├── AXA-CAR-PLUS.pdf
├── AVIVA-SHORT-TERM.pdf
├── manifest.json
└── README.md

## Funcionamiento

Durante la indexación, InsurGuard AI:

1. Lee cada póliza PDF.
2. Extrae directamente el texto de las páginas digitales.
3. Cuando una página no contiene texto utilizable, aplica OCR mediante
   Tesseract exclusivamente sobre esa página.
4. Divide el contenido en fragmentos.
5. Genera embeddings mediante Sentence Transformers.
6. Almacena los embeddings en PostgreSQL utilizando pgvector.

Posteriormente, el supervisor puede realizar preguntas sobre una póliza desde
la consola administrativa y consultar el fragmento y la página exacta utilizados
como evidencia.

## Reindexación

Después de copiar las pólizas en esta carpeta se debe ejecutar:

```bash
python scripts/index_policies.py --force