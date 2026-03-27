# Reporte de Modernización del Código — OpenChem

> **Contexto:** Este repositorio fue originalmente diseñado para Python 3.6/3.7.
> El entorno actual de ejecución utiliza Python 3.12.  
> Este reporte identifica patrones modernizables **sin proponer cambios al código de producción**:
> solo documenta qué podría actualizarse, con qué prioridad y qué riesgos conlleva cada cambio.

---

## Índice

1. [Bugs críticos activos (no son solo modernización)](#1-bugs-críticos-activos)
2. [Dependencias obsoletas o incompatibles](#2-dependencias-obsoletas-o-incompatibles)
3. [Compatibilidad Python 2 residual](#3-compatibilidad-python-2-residual)
4. [Patrón `super()` antiguo](#4-patrón-super-antiguo)
5. [Apertura de archivos sin context manager](#5-apertura-de-archivos-sin-context-manager)
6. [Manejo de excepciones inseguro](#6-manejo-de-excepciones-inseguro)
7. [Verificación de tipos con `type() ==`](#7-verificación-de-tipos-con-type-)
8. [Formateo de strings antiguo](#8-formateo-de-strings-antiguo)
9. [Falta de type hints](#9-falta-de-type-hints)
10. [Patrones menores y estilo](#10-patrones-menores-y-estilo)
11. [Infraestructura CI/Docker desactualizada](#11-infraestructura-cidocker-desactualizada)
12. [Deuda técnica documentada (TODO/FIXME/BUG)](#12-deuda-técnica-documentada-todofixmebug)
13. [Resumen de prioridades](#resumen-de-prioridades)

---

## 1. Bugs Críticos Activos

> ⚠️ Estos no son solo oportunidades de modernización: son bugs reales que pueden causar fallos silenciosos o crashes en tiempo de ejecución.

### 1.1 Nombre de excepción inexistente: `ArgumentError`

**Archivo:** `openchem/data/siamese_data_layer.py`, líneas 29 y 40

```python
# Código actual (líneas 29 y 40):
else:
    raise ArgumentError
```

`ArgumentError` no existe en Python built-in ni está importado en el módulo. Al ejecutarse esta rama, Python lanza un `NameError: name 'ArgumentError' is not defined` en lugar del error esperado. La intención era probablemente `TypeError` o `ValueError`.

**Riesgo de la corrección:** 🟢 **Inocuo.** El código actual ya falla con `NameError`; corregirlo a `TypeError` o `ValueError` solo mejora el mensaje de error.

---

### 1.2 Typo en nombre de excepción: `ValuError`

**Archivo:** `openchem/modules/encoders/cnn_encoder.py`, línea 23

```python
# Código actual:
raise ValuError("Pooling must be one of 'max', 'mean', 'sum'")
```

`ValuError` (sin la `e`) no existe. Python lanza `NameError: name 'ValuError' is not defined` en lugar del `ValueError` esperado. La validación de `pooling` nunca funciona correctamente.

**Riesgo de la corrección:** 🟢 **Inocuo.**

---

### 1.3 Typo en keyword argument: `for_predction`

**Archivo:** `openchem/models/openchem_model.py`, línea 321

```python
# Código actual (rama `else` de predict()):
batch_input, batch_object = model.cast_inputs(sample_batched,
                                              task,
                                              use_cuda,
                                              for_predction=True)  # typo
```

La firma correcta del parámetro es `for_prediction`. Como `cast_inputs` acepta `**kwargs` implícitamente en algunos modelos, este typo puede pasar desapercibido hasta tiempo de ejecución, donde `for_prediction` quedará en `False` (su valor por defecto) silenciosamente.

La línea 314 (rama `if`) usa correctamente `for_prediction=True`.

**Riesgo de la corrección:** 🟡 **Bajo.** Corregirlo activa la lógica de predicción que antes estaba silenciosamente deshabilitada. Podría cambiar el comportamiento del modo `predict` en modelos sin módulo wrapper.

---

### 1.4 Archivo no cerrado en `evaluate()` — fuga de recurso

**Archivo:** `openchem/models/openchem_model.py`, líneas 276–283

```python
# Código actual (simplificado):
if task == "graph_generation":
    f = open(logdir + "/debug_smiles_epoch_" + str(epoch) + ".smi", "w")
    if isinstance(metrics, list) and len(metrics) == len(prediction):
        for i in range(len(prediction)):
            f.writelines(str(prediction[i]) + "," + str(metrics[i]) + "\n")
    else:
        for i in range(len(prediction)):
            f.writelines(str(prediction[i]) + "\n")
        f.close()  # ← solo se cierra en el else
# f nunca se cierra en la rama if del isinstance
```

En la rama `if isinstance(metrics, list)`, el archivo `f` **nunca se cierra**. Esto es una fuga de descriptor de archivo.

**Riesgo de la corrección:** 🟢 **Inocuo** (usar `with open(...)` soluciona ambas ramas).

---

## 2. Dependencias Obsoletas o Incompatibles

### 2.1 `sklearn.externals.joblib` — eliminado desde scikit-learn 0.23

**Archivo:** `openchem/models/vanilla_model.py`, línea 10

```python
from sklearn.externals import joblib
```

`sklearn.externals.joblib` fue marcado como deprecated en sklearn 0.21 y **eliminado completamente en sklearn 0.23** (2020). En cualquier entorno moderno este import falla con `ImportError`. El módulo correcto es:

```python
import joblib
```

**Riesgo de la corrección:** 🟢 **Inocuo.** `joblib` es ahora un paquete independiente; la API es idéntica.

---

### 2.2 TensorFlow 1.x API — incompatible con TF 2.x

**Archivo:** `openchem/utils/logger.py`

```python
import tensorflow as tf
# ...
self.writer = tf.summary.FileWriter(log_dir)   # TF 1.x
summary = tf.Summary(value=[...])              # TF 1.x
tf.Summary.Value(tag=tag, simple_value=value)  # TF 1.x
tf.HistogramProto()                            # TF 1.x
```

Toda la clase `Logger` usa la API de TensorFlow 1.x (`tf.summary.FileWriter`, `tf.Summary`, `tf.HistogramProto`), que **no existe en TensorFlow 2.x**. Desde TF 2.0 (septiembre 2019), la API de summaries cambió a `tf.summary.create_file_writer()`.

Adicionalmente, la misma clase importa `scipy.misc.toimage` (línea 5), que fue **eliminado en SciPy 1.3.0** (2019). El reemplazo es `PIL.Image.fromarray()`.

También tiene compatibilidad Python 2 residual:
```python
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
```

**Riesgo de la corrección:** 🔴 **Alto.** Reescribir `Logger` implica comprender el flujo completo de summaries y afecta el logging de entrenamiento. Sin embargo, nótese que `logger.py` **no parece usarse actualmente** en el código principal (la integración actual usa `torch.utils.tensorboard.SummaryWriter` en `openchem_model.py`). El módulo podría ser **código muerto**.

---

### 2.3 `six` — librería de compatibilidad Python 2/3 innecesaria

**Archivos:** `openchem/utils/utils.py` (líneas 3, 9, 43–45) y `run.py` (línea 13)

```python
# utils.py
from six import string_types
import six
# ...
if six.PY2:
    print((start + " " * offset + line).encode('utf-8'), end=end)
else:
    print(start + " " * offset + line, end=end)

# run.py
from six import string_types
```

La librería `six` existe exclusivamente para dar compatibilidad entre Python 2 y Python 3. En Python 3:
- `six.PY2` siempre es `False`; la rama muerta nunca se ejecuta.
- `string_types` es simplemente `(str,)`; puede reemplazarse con `str`.

**Riesgo de la corrección:** 🟢 **Inocuo** en Python 3 puro. Remover `six` y reemplazar `string_types` por `str` no cambia el comportamiento.

---

### 2.4 Configuración de `setup.py` desactualizada

**Archivo:** `setup.py`

```python
'python_requires': ">=3.5",   # muy permisivo para código que usa f-strings implícitos
'use_scm_version': True,
'setup_requires': ['setuptools_scm'],
```

El formato moderno de packaging usa `pyproject.toml` (PEP 517/518/621). El uso de `setup.py` como punto de entrada principal quedó deprecated a partir de setuptools 61+ (2022). La configuración `use_scm_version` sin `version_file` puede provocar que el paquete se instale sin versión en entornos sin git.

**Riesgo de la corrección:** 🟡 **Medio.** Migrar a `pyproject.toml` requiere reorganizar la metadata del paquete. No cambia funcionalidad, pero es un cambio estructural.

---

## 3. Compatibilidad Python 2 Residual

### 3.1 `__future__` imports innecesarios

**Archivo:** `openchem/models/vanilla_model.py`, líneas 1–2

```python
from __future__ import print_function
from __future__ import division
```

Estos imports existían para que código Python 2 tuviera las semánticas de Python 3. En Python 3, `print` ya es función y la división ya es verdadera (`/` retorna float). Son completamente inofensivos pero indican que el código fue escrito o portado desde Python 2.

**Riesgo de la corrección:** 🟢 **Inocuo.** Eliminarlos no cambia nada.

---

### 3.2 Compatibilidad `StringIO` en `logger.py`

**Archivo:** `openchem/utils/logger.py`, líneas 5–8

```python
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
```

`StringIO` (sin módulo `io`) no existe en Python 3. Este bloque siempre toma el camino `except`. En Python 3, la import correcta es `from io import BytesIO`.

**Riesgo de la corrección:** 🟢 **Inocuo.**

---

## 4. Patrón `super()` Antiguo

**Alcance:** 38 archivos, ~40 instancias

En Python 3, `super()` sin argumentos es equivalente a `super(ClassName, self)`. El patrón con argumentos es un vestigio de Python 2.

**Ejemplos representativos:**

| Archivo | Línea | Código actual |
|---------|-------|--------------|
| `openchem/models/openchem_model.py` | 27 | `super(OpenChemModel, self).__init__()` |
| `openchem/data/utils.py` | 25 | `super(DummyDataset, self).__init__()` |
| `openchem/layers/gcn.py` | 16 | `super(GraphConvolution, self).__init__()` |
| `openchem/modules/mlp/openchem_mlp.py` | 12, 67 | `super(OpenChemMLP, self).__init__()` |
| `openchem/models/MolecularRNN.py` | 28, 471 | `super(MolecularRNNModel, self).__init__(params)` |

Lista completa de archivos afectados:
- `criterion/multitask_loss.py`, `criterion/policy_gradient_loss.py`
- `data/feature_data_layer.py`, `data/graph_data_layer.py`, `data/siamese_data_layer.py`
- `data/smiles_data_layer.py`, `data/smiles_enumerator.py`, `data/smiles_protein_data_layer.py`
- `data/utils.py`, `data/vanilla_data_layer.py`
- `layers/conv_bn_relu.py`, `layers/gcn.py`, `layers/stack_augmentation.py`
- `models/GenerativeRNN.py`, `models/Graph2Label.py`, `models/MLP2Label.py`
- `models/MolecularRNN.py`, `models/MoleculeProtein2Label.py`, `models/SiameseModel.py`
- `models/Smiles2Label.py`, `models/openchem_model.py`, `models/vanilla_model.py`
- `modules/embeddings/basic_embedding.py`, `modules/embeddings/onehot_embedding.py`
- `modules/embeddings/openchem_embedding.py`
- `modules/encoders/cnn_encoder.py`, `modules/encoders/edge_attention_encoder.py`
- `modules/encoders/gcn_encoder.py`, `modules/encoders/openchem_encoder.py`
- `modules/encoders/rnn_encoder.py`
- `modules/gru_plain.py`, `modules/mlp/openchem_mlp.py`

**Modernización:**
```python
# Actual (Python 2 style)
super(OpenChemModel, self).__init__()

# Moderno (Python 3)
super().__init__()
```

**Riesgo de la corrección:** 🟢 **Inocuo en la gran mayoría de casos.** La única excepción es cuando hay herencia múltiple compleja con MRO no trivial, pero ninguna clase del proyecto parece usarla de esa manera. Se puede automatizar con la herramienta `pyupgrade`.

---

## 5. Apertura de Archivos sin Context Manager

**Archivos afectados:** `data/utils.py`, `data/graph_data_layer.py`, `data/smiles_protein_data_layer.py`, `models/openchem_model.py`, `utils/sa_score/sascorer.py`

El patrón `f = open(...)` sin `with` puede dejar archivos abiertos si ocurre una excepción antes de la llamada a `f.close()`.

| Archivo | Línea | Descripción |
|---------|-------|-------------|
| `data/graph_data_layer.py` | 52 | `pickle.load(open(filename, "rb"))` — sin `with`, sin cierre |
| `data/utils.py` | 249 | `f = open(filename, 'w')` — cierra en 252 |
| `data/utils.py` | 269 | `f = open(filename, 'r')` — cierra en 277 |
| `data/utils.py` | 325 | `csv.reader(open(path, 'r'), ...)` — nunca se cierra explícitamente |
| `data/utils.py` | 340 | `f = open(path, 'w')` — cierra en 347 |
| `data/smiles_protein_data_layer.py` | 37 | `f = open(filename, 'rb')` |
| `models/openchem_model.py` | 276 | `f = open(...)` — no se cierra en rama `if` (ver §1.4) |
| `models/openchem_model.py` | 330 | `f = open(logdir + "/predictions.txt", "w")` |
| `utils/sa_score/sascorer.py` | 37 | `gzip.open('%s.pkl.gz' % name)` sin `with` |

**Modernización:**
```python
# Actual
f = open(filename, 'w')
f.writelines(data)
f.close()

# Moderno
with open(filename, 'w') as f:
    f.writelines(data)
```

**Riesgo de la corrección:** 🟢 **Inocuo.** El uso de `with` es funcionalmente equivalente y más robusto. Herramienta: `pyupgrade` o refactoring manual.

---

## 6. Manejo de Excepciones Inseguro

### 6.1 `except:` desnudo (captura todo, incluyendo `KeyboardInterrupt` y `SystemExit`)

**Archivo:** `openchem/data/utils.py`, línea 223

```python
try:
    new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
    idx.append(i)
except:          # ← captura CUALQUIER excepción, incluyendo Ctrl+C
    new_smiles.append('')
```

Un `except:` desnudo captura `BaseException`, incluyendo `KeyboardInterrupt`, `SystemExit` y `GeneratorExit`. Esto puede hacer que el programa sea imposible de interrumpir o que errores de programación (como `MemoryError`) queden silenciados.

**Archivo:** `openchem/utils/graph.py`, línea 144

```python
except:
    raise ValueError(...)
```

Aquí se captura cualquier error y se re-lanza como `ValueError`, perdiendo el traceback original y el tipo de excepción.

**Modernización:**
```python
except Exception:    # solo excepciones de aplicación
    new_smiles.append('')
```

**Riesgo de la corrección:** 🟡 **Bajo-Medio.** Cambiar a `except Exception` puede exponer errores que antes eran silenciados (especialmente en la sanitización de SMILES). Requiere validar que el comportamiento sigue siendo correcto con moléculas inválidas.

---

## 7. Verificación de Tipos con `type() ==`

### 7.1 `type(x) == SomeType` en lugar de `isinstance()`

**Archivo:** `run.py`, líneas 96–99

```python
if type(value) == int or type(value) == float or isinstance(value, string_types):
    parser_unk.add_argument(...)
elif type(value) == bool:
    parser_unk.add_argument(...)
```

**Archivo:** `openchem/modules/mlp/openchem_mlp.py`, líneas 19 y 74

```python
if type(self.activation) is list:
```

El uso de `type(x) == T` no detecta subclases y es considerado un anti-patrón. `isinstance(x, T)` es el idioma correcto. Nótese también que en `run.py` la verificación de `bool` debe ir **antes** que la de `int` porque `bool` es subclase de `int`.

**Modernización:**
```python
# Actual
if type(value) == int or type(value) == float or isinstance(value, string_types):
    ...
elif type(value) == bool:

# Moderno
if isinstance(value, bool):      # ← bool primero (es subclase de int)
    ...
elif isinstance(value, (int, float, str)):
    ...
```

**Riesgo de la corrección:** 🟡 **Medio.** En `run.py`, el orden actual de las verificaciones pone `bool` en el `elif`, pero `type(True) == int` es `False` (porque `type()` no sigue MRO). Irónicamente, el código actual funciona para `bool` gracias al `elif`, pero es frágil y confuso. Cambiar a `isinstance` requiere reordenar las condiciones, lo que puede alterar el comportamiento si no se hace con cuidado.

---

## 8. Formateo de Strings Antiguo

Python 3.6+ introdujo f-strings, que son más legibles, más rápidas y menos propensas a errores que `%` o `.format()`.

### 8.1 Formato `%` (estilo C/Python 2)

| Archivo | Línea | Código |
|---------|-------|--------|
| `openchem/utils/utils.py` | 97 | `'%dm %ds' % (m, s)` |
| `openchem/data/utils.py` | 228 | `'Proportion of unsanitized smiles is %.3f ' % (invalid_rate)` |
| `openchem/data/utils.py` | 321 | `'%dm %ds' % (m, s)` |
| `openchem/data/smiles_enumerator.py` | 77 | `'Found: X.shape = %s, y.shape = %s' % (...)` |
| `openchem/data/smiles_enumerator.py` | 228 | `"Error in reconstruction %s %s" % (smile, smiles[i])` |
| `openchem/models/openchem_model.py` | 171 | `'TRAINING: [Time: %s, Epoch: %d, ...]' % (...)` |
| `openchem/models/openchem_model.py` | 286 | `'EVALUATION: [Time: %s, ...]' % (...)` |
| `openchem/models/openchem_model.py` | 347 | `'PREDICTION: [Time: %s, ...]' % (...)` |
| `openchem/utils/logger.py` | 42 | `'%s/%d' % (tag, i)` |
| `openchem/utils/sa_score/sascorer.py` | 139 | `'Reading took %.2f seconds...' % (...)` |

### 8.2 Formato `.format()` (Python 3.0+, reemplazable por f-strings)

Más de 30 instancias en `run.py`, `openchem/utils/utils.py`, `openchem/data/utils.py`, `openchem/models/MolecularRNN.py`, `openchem/models/openchem_model.py`.

**Modernización (ejemplo):**
```python
# Formato %
'TRAINING: [Time: %s, Epoch: %d, Progress: %d%%, Loss: %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, cur_loss)

# Formato .format()
'TRAINING: [Time: {}, Epoch: {}, Progress: {:.0%}, Loss: {:.4f}]'.format(time_since(start), epoch, epoch / n_epochs, cur_loss)

# f-string (Python 3.6+)
f'TRAINING: [Time: {time_since(start)}, Epoch: {epoch}, Progress: {epoch / n_epochs:.0%}, Loss: {cur_loss:.4f}]'
```

**Riesgo de la corrección:** 🟢 **Inocuo** en la mayoría de los casos. Los f-strings no cambian la semántica. La única precaución es con los especificadores de formato que difieren ligeramente entre `%` y f-strings (ej. `%d` vs `:d`, `%%` vs `%`). Se puede automatizar con `pyupgrade --py36-plus`.

---

## 9. Falta de Type Hints

**Alcance:** Todos los archivos del proyecto.

El proyecto no tiene ninguna anotación de tipos. Para Python 3.5+ esto puede añadirse progresivamente sin romper nada.

**Prioridades sugeridas para anotación:**

1. **Interfaces públicas:** `sanitize_smiles`, `create_loader`, `seq2tensor`, `pad_sequences` en `data/utils.py`
2. **Clases base:** `OpenChemModel`, `OpenChemEncoder`, `OpenChemEmbedding`
3. **Funciones utilitarias:** `flatten_dict`, `nested_update`, `check_params` en `utils/utils.py`

**Ejemplo de modernización:**
```python
# Actual
def sanitize_smiles(smiles, canonize=True, min_atoms=-1, max_atoms=-1,
                    return_num_atoms=False, allowed_tokens=None,
                    allow_charges=False, return_max_len=False, logging="warn"):

# Con type hints (Python 3.9+)
def sanitize_smiles(
    smiles: list[str],
    canonize: bool = True,
    min_atoms: int = -1,
    max_atoms: int = -1,
    return_num_atoms: bool = False,
    allowed_tokens: list[str] | None = None,
    allow_charges: bool = False,
    return_max_len: bool = False,
    logging: str = "warn"
) -> tuple[list[str], list[int]]:
```

**Riesgo de la corrección:** 🟢 **Inocuo.** Las anotaciones de tipo son ignoradas en tiempo de ejecución por Python (no son validadas automáticamente). No cambian el comportamiento del programa. Se pueden añadir gradualmente con herramientas como `mypy` para validación estática.

> **Nota:** Para compatibilidad con Python 3.7–3.8, usar `from __future__ import annotations` o los tipos del módulo `typing` (`List`, `Dict`, `Optional`, etc.).  
> Para Python 3.9+, los built-ins `list`, `dict`, `tuple` se pueden usar directamente como genéricos.  
> Para Python 3.10+, la sintaxis `X | Y` reemplaza `Union[X, Y]`.

---

## 10. Patrones Menores y Estilo

### 10.1 `while 1:` en lugar de `while True:`

**Archivo:** `openchem/data/smiles_enumerator.py`, línea 33

```python
while 1:    # idioma de Python 2
    ...

# Moderno
while True:
```

**Riesgo:** 🟢 **Inocuo.** En Python 3 ambas formas son equivalentes en rendimiento (el compilador las optimiza igual).

---

### 10.2 Clases de datos sin decorador `@dataclass`

**Archivo:** `openchem/utils/graph.py`

Las clases `Attribute`, `Node`, `Edge` y `Graph` son esencialmente contenedores de datos con `__init__` manual. Python 3.7 introdujo `@dataclass` que genera `__init__`, `__repr__` y `__eq__` automáticamente.

```python
# Actual
class Node:
    def __init__(self, idx, rdmol, get_atom_attributes, has_3D=False):
        rdatom = rdmol.GetAtoms()[idx]
        self.node_idx = idx
        self.atom_type = rdatom.GetAtomicNum()
        ...

# Con dataclass (solo si la lógica del __init__ es simple)
from dataclasses import dataclass, field

@dataclass
class NodeData:
    node_idx: int
    atom_type: int
    attributes_dict: dict
```

**Riesgo:** 🔴 **Alto para `Node` y `Graph`.** Sus `__init__` tienen lógica compleja (acceden a `rdmol`, calculan atributos, construyen matrices de adyacencia). Convertirlos directamente a dataclasses requeriría mover esa lógica a `__post_init__`, lo cual es factible pero no trivial. **No se recomienda** sin tests comprehensivos.

---

### 10.3 `os.path` vs `pathlib.Path`

**Archivos:** `run.py` (10+ usos), `utils/utils.py` (3 usos), `utils/sa_score/sascorer.py` (2 usos)

`pathlib.Path` (Python 3.4+) ofrece una API orientada a objetos más expresiva que `os.path`.

```python
# Actual
ckpt_dir = os.path.join(logdir, 'checkpoint')
checkpoint = os.path.basename(checkpoint).split("_")[-1]

# Con pathlib
from pathlib import Path
ckpt_dir = Path(logdir) / 'checkpoint'
checkpoint = Path(checkpoint).stem.split("_")[-1]
```

**Riesgo:** 🟡 **Bajo-Medio.** `pathlib.Path` y `str` no son intercambiables en todos los contextos. Algunas APIs de terceros (pytorch, rdkit) pueden requerir strings explícitos. El cambio debe hacerse verificando cada llamada downstream.

---

### 10.4 Ausencia de `__all__` en módulos

Ningún módulo del paquete `openchem` define `__all__`. Esto significa que `from openchem.data.utils import *` exportaría todos los nombres públicos incluyendo imports intermedios.

**Riesgo de la corrección:** 🟢 **Inocuo.** Solo afecta a `import *`, que no se usa en el codebase.

---

### 10.5 String de mezcla innecesaria (concatenación literal)

**Archivo:** `openchem/utils/utils.py`, línea 84

```python
raise ValueError("Mismatch between org_dict and upd_dict " "at node {}".format(key))
```

Dos string literals adyacentes se concatenan en tiempo de compilación. Es válido pero confuso; sería más claro como una sola string.

---

### 10.6 `for i in range(len(x))` en lugar de `enumerate`

Ejemplos en varios archivos:

```python
# Patrón antiguo
for i in range(len(seqs)):
    cur_len = len(seqs[i])
    seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)

# Moderno
for i, seq in enumerate(seqs):
    seqs[i] = seq + pad_symbol * (max_length - len(seq))
```

**Archivos con este patrón:** `data/utils.py`, `data/smiles_enumerator.py`, `models/openchem_model.py`

**Riesgo:** 🟢 **Inocuo** si el índice se usa solo para leer. Si se usa para mutación in-place, `enumerate` también es válido pero requiere atención.

---

### 10.7 `assert` para validación de precondiciones de producción

**Archivos:** `models/vanilla_model.py`, `data/utils.py`, `models/openchem_model.py`

```python
assert len(smiles) == len(fps)
assert len(clean_smiles) == len(prediction)
```

Los `assert` se deshabilitan con la bandera `-O` (optimize) de Python. Para validaciones de producción, es preferible usar `if ... raise ValueError(...)`.

**Riesgo de la corrección:** 🟡 **Bajo.** El código no parece ejecutarse en modo optimizado, pero es una mala práctica.

---

## 11. Infraestructura CI/Docker Desactualizada

### 11.1 `.travis.yml` — Python 3.6/3.7 solamente

```yaml
# .travis.yml
env: PYTHON_VER=3.6
env: PYTHON_VER=3.7
```

El CI solo prueba Python 3.6 y 3.7, ambas versiones con **soporte terminado** (EOL). Python 3.6 llegó a EOL en diciembre 2021, Python 3.7 en junio 2023. La versión mínima soportada actualmente es Python 3.8.

**Riesgo de la corrección:** 🟡 **Medio.** Actualizar el CI a Python 3.10+ puede revelar incompatibilidades previamente no detectadas (incluyendo varios de los problemas documentados en este reporte).

---

### 11.2 `Dockerfile` — CUDA 9.0, Ubuntu 16.04

```dockerfile
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
```

CUDA 9.0 fue lanzado en 2017; las GPUs modernas (Ampere, Hopper, Ada Lovelace) requieren CUDA 11+. Ubuntu 16.04 llegó a EOL en abril 2021.

**Riesgo de la corrección:** 🔴 **Alto.** Actualizar la imagen base puede requerir cambios en las dependencias de PyTorch, el proceso de compilación y posiblemente en el código de operaciones CUDA.

---

### 11.3 `requirements.txt` sin versiones pinneadas

```
numpy
pyyaml
scipy
# ...
```

Ningún paquete tiene versión especificada, lo que puede causar incompatibilidades silenciosas con versiones futuras de dependencias.

**Riesgo de la corrección:** 🟡 **Medio.** Pinnear versiones mejora reproducibilidad pero requiere mantenerlas actualizadas.

---

## 12. Deuda Técnica Documentada (TODO/FIXME/BUG)

El código contiene 17 comentarios de deuda técnica activa:

| Archivo | Línea | Tipo | Descripción |
|---------|-------|------|-------------|
| `data/utils.py` | 1 | `TODO` | packed variable length sequence |
| `data/smiles_data_layer.py` | 1 | `TODO` | packed variable length sequence |
| `data/smiles_protein_data_layer.py` | 1 | `TODO` | packed variable length sequence |
| `data/graph_data_layer.py` | 1 | `TODO` | variable length batching |
| `data/graph_data_layer.py` | 118 | `TODO` | remove diagonal elements from adjacency matrix |
| `data/graph_data_layer.py` | 273 | `TODO` | remove constant 1008 from here |
| `data/graph_data_layer.py` | 283 | `TODO` | is copy needed here? |
| `data/graph_data_layer.py` | 290 | `TODO` | the first input token is all ones? |
| `data/utils.py` | 396 | `TODO` | remove diagonal elements from adjacency matrix |
| `data/smiles_enumerator.py` | 238 | `BUG` | when batchsize > x.shape[0], returns x.shape[0] only |
| `models/MolecularRNN.py` | 60 | `TODO` | rewrite in OpenChem native style |
| `models/MolecularRNN.py` | 96 | `TODO` | implement required params |
| `models/MolecularRNN.py` | 102 | `TODO` | check if .eval() creates problems with batchnorm/dropout |
| `models/MolecularRNN.py` | 130 | `TODO` | handle float type for x_step (no node embedding) |
| `models/MolecularRNN.py` | 270 | `TODO` | avoid double sanitization |
| `models/MolecularRNN.py` | 359 | `TODO` | consistent with BFSGraphDataset |
| `models/MolecularRNN.py` | 472 | `TODO` | load from original checkpoint if fails |
| `modules/gru_plain.py` | 27 | `TODO` | use small embedding layer for edge class |
| `modules/encoders/rnn_encoder.py` | 77 | `TODO` | output shape changed (batch_first=True), check hidden |

El `BUG` en `smiles_enumerator.py:238` es especialmente notable: está documentado explícitamente en el código como un bug conocido sin solución.

---

## Resumen de Prioridades

### 🔴 Crítico — Bugs que ya causan fallos

| ID | Descripción | Archivo | Riesgo al corregir |
|----|-------------|---------|-------------------|
| 1.1 | `ArgumentError` no definido | `siamese_data_layer.py:29,40` | 🟢 Inocuo |
| 1.2 | Typo `ValuError` | `cnn_encoder.py:23` | 🟢 Inocuo |
| 1.3 | Typo `for_predction` (kwarg silencioso) | `openchem_model.py:321` | 🟡 Bajo |
| 1.4 | Archivo no cerrado en `evaluate()` | `openchem_model.py:276` | 🟢 Inocuo |
| 2.1 | `sklearn.externals.joblib` eliminado | `vanilla_model.py:10` | 🟢 Inocuo |

### 🟠 Alto — Incompatibilidades en entornos modernos

| ID | Descripción | Archivos | Riesgo al corregir |
|----|-------------|---------|-------------------|
| 2.2 | TF 1.x API (+ scipy.misc.toimage eliminado) | `logger.py` | 🔴 Alto (reescritura) |
| 11.2 | Dockerfile con CUDA 9.0 y Ubuntu 16.04 | `Dockerfile` | 🔴 Alto |

### 🟡 Medio — Modernización con bajo riesgo

| ID | Descripción | Alcance | Riesgo al corregir |
|----|-------------|---------|-------------------|
| 2.3 | Librería `six` innecesaria | 2 archivos | 🟢 Inocuo |
| 3.1 | `__future__` imports | `vanilla_model.py` | 🟢 Inocuo |
| 4 | Patrón `super()` antiguo | ~40 instancias | 🟢 Inocuo (automatizable) |
| 5 | `open()` sin context manager | 8 instancias | 🟢 Inocuo |
| 6 | `except:` desnudo | 2 instancias | 🟡 Bajo |
| 7 | `type() ==` en lugar de `isinstance()` | 4 instancias | 🟡 Medio |
| 8 | Formato `%` y `.format()` | ~40 instancias | 🟢 Inocuo (automatizable) |
| 11.1 | CI con Python 3.6/3.7 EOL | `.travis.yml` | 🟡 Medio |

### 🟢 Bajo — Mejoras de calidad sin urgencia

| ID | Descripción | Riesgo al corregir |
|----|-------------|-------------------|
| 9 | Falta de type hints | 🟢 Inocuo |
| 10.1 | `while 1:` → `while True:` | 🟢 Inocuo |
| 10.3 | `os.path` → `pathlib` | 🟡 Bajo |
| 10.4 | Ausencia de `__all__` | 🟢 Inocuo |
| 10.6 | `range(len(x))` → `enumerate` | 🟢 Inocuo |
| 10.7 | `assert` para validaciones de producción | 🟡 Bajo |
| 11.3 | `requirements.txt` sin versiones | 🟡 Medio |
| 12 | Deuda técnica TODO/BUG activa | Variable |

---

## Herramientas Recomendadas para Automatización

Si se decide proceder con la modernización, las siguientes herramientas permiten automatizar los cambios de bajo riesgo:

```bash
# pyupgrade: moderniza sintaxis automáticamente
pip install pyupgrade
pyupgrade --py39-plus openchem/**/*.py

# isort: organiza imports
pip install isort
isort openchem/

# black: formateo de código
pip install black
black openchem/

# mypy: verificación de tipos
pip install mypy
mypy openchem/ --ignore-missing-imports

# flake8: linting general
pip install flake8
flake8 openchem/
```

> `pyupgrade --py39-plus` puede automatizar: `super()`, `%`-format básico, `__future__` imports, `Union[X, Y]` → `X | Y`, tipos genéricos built-in.

---

*Reporte generado el 2026-03-27. Analizados 48 archivos Python en el repositorio OpenChem.*
