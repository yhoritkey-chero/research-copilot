from openai import OpenAI

PROMPT_V1 = """\
Eres un asistente de investigacion estrictamente limitado al CONTEXTO.
NO puedes usar tu conocimiento previo. NO menciones años (como 2021) o autores que no esten en el contexto.

### CONTEXTO ###
{context}

### PREGUNTA ###
{question}

### INSTRUCCIONES ###
- Responde solo usando los fragmentos numerados.
- Si no hay informacion exacta, di que no lo sabes.
- Cita los titulos reales de los papers.
- Maximo 3 parrafos."""

PROMPT_V2 = """\
Eres un asistente de investigacion. Responde basandote SOLO en el contexto.
PROHIBIDO inventar años o autores (ej. no menciones 2021 si no esta arriba).

Devuelve tu respuesta UNICAMENTE como un objeto JSON:
{{
  "respuesta": "respuesta basada solo en fragmentos numerados aqui",
  "puntos_clave": ["punto 1", "punto 2"],
  "fuentes_verificadas": ["titulo real del fragmento"],
  "nivel_confianza": "bajo si tienes que adivinar algo"
}}

CONTEXTO:
{context}

PREGUNTA: {question}"""

PROMPT_V3 = """\
Eres un asistente de investigacion que responde preguntas sobre papers de regulacion de IA.

EJEMPLO 1:
Pregunta: Que establece el AI Act de la UE sobre sistemas de alto riesgo?
Respuesta: El AI Act establece un marco basado en riesgo donde los sistemas de IA de alto
riesgo deben cumplir requisitos estrictos: gestion de riesgos, gobernanza de datos,
transparencia y supervision humana. Desarrolladores y desplegadores tienen obligaciones
diferenciadas segun su rol en la cadena de valor (Hacker et al., 2023).

EJEMPLO 2:
Pregunta: Como se relaciona el GDPR con la toma de decisiones automatizada?
Respuesta: El Articulo 22 del GDPR prohibe decisiones basadas unicamente en procesamiento
automatizado con efectos significativos. Sin embargo, contiene numerosas excepciones.
Los responsables deben proveer informacion significativa sobre la logica involucrada (Brkan, 2017).

Ahora responde usando el siguiente contexto:

CONTEXTO:
{context}

Pregunta: {question}
Respuesta:"""

PROMPT_V4 = """\
Eres un asistente de investigacion estrictamente limitado al CONTEXTO proporcionado.
TU CONOCIMIENTO PREVIO NO EXISTE. Solo puedes usar la información que aparezca debajo entre [1] y [n].

### CONTEXTO SUMINISTRADO ###
{context}

### PREGUNTA DEL USUARIO ###
{question}

### REGLAS CRÍTICAS DE SEGURIDAD ###
1. **PROHIBIDO** mencionar autores o papers que NO estén en la lista numerada de arriba (ej. NO menciones a Smith, Johnson o European Commission si no tienen un número asignado arriba).
2. **PROHIBIDO** inventar información. Si el contexto no contiene la respuesta, di: "No hay información suficiente en los documentos cargados para responder esta pregunta".
3. **CITA OBLIGATORIA**: Cada frase de tu respuesta debe terminar con el número del fragmento de donde salió (ej. [1]).
4. **VERIFICACIÓN DE FUENTES**: Al final de tu respuesta, haz una lista de "Fuentes Reales Utilizadas" citando solo los títulos tal cual aparecen arriba.

### ESTRUCTURA OBLIGATORIA DE RESPUESTA ###
Analiza paso a paso:
1. Fragmentos encontrados: (Enlista solo los números que sirven)
2. Síntesis Basada en Evidencia: (Responde usando citas [x] en cada punto)
3. Fuentes Reales Utilizadas: (Copia los títulos de los fragmentos arriba)"""



STRATEGIES = {'v1': PROMPT_V1, 'v2': PROMPT_V2, 'v3': PROMPT_V3, 'v4': PROMPT_V4}

_client = OpenAI()


def generate(question, context, strategy='v1'):
    prompt = STRATEGIES[strategy].format(context=context, question=question)
    response = _client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content
