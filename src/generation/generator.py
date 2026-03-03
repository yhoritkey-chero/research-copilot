from openai import OpenAI

PROMPT_V1 = """\
Eres un asistente de investigacion especializado en regulacion de IA y privacidad de datos.

### CONTEXTO (fragmentos de papers academicos) ###
{context}

### PREGUNTA ###
{question}

### INSTRUCCIONES ###
- Responde UNICAMENTE basandote en el contexto proporcionado.
- Cita los titulos de los papers cuando uses informacion especifica.
- Si el contexto no contiene suficiente informacion, indicalo.
- Usa lenguaje academico y preciso.
- Maximo 3 parrafos."""

PROMPT_V2 = """\
Eres un asistente de investigacion. Responde la pregunta basandote en el contexto.

Devuelve tu respuesta UNICAMENTE como un objeto JSON con esta estructura:
{{
  "respuesta": "respuesta principal aqui",
  "puntos_clave": ["punto 1", "punto 2", "punto 3"],
  "fuentes": ["titulo paper 1", "titulo paper 2"],
  "nivel_confianza": "alto/medio/bajo",
  "limitaciones": "que informacion falta o es incierta"
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
Eres un asistente de investigacion experto. Piensa paso a paso para responder la pregunta basándote únicamente en el CONTEXTO proporcionado.

### CONTEXTO ###
{context}

### PREGUNTA ###
{question}

### CRITERIOS DE RESPUESTA ###
1. Analiza el contexto fragmento por fragmento.
2. Identifica los papers específicos que responden a la pregunta.
3. Para cada afirmación, utiliza citas numéricas (ej. [1], [2]) correspondientes a los fragmentos del contexto.
4. Si varios papers coinciden, cita todos los números relevantes.
5. NO inventes nombres de papers como "Paper A" o "Paper B". Usa los títulos reales que aparecen en el contexto.

### ESTRUCTURA DE LA RESPUESTA ###
Razona paso a paso:
1. Análisis de la pregunta.
2. Identificación de evidencia en los fragmentos (cita los números [x]).
3. Síntesis de perspectivas.
4. Respuesta final con citas integradas (estilo: "Según el paper X [1], ocurre Y...")."""


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
