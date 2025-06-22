# Guía de Contribución a Med-STORM

¡Gracias por tu interés en contribuir a Med-STORM! Apreciamos mucho tu tiempo y esfuerzo. Por favor, tómate un momento para revisar esta guía antes de enviar tus contribuciones.

## Cómo Contribuir

1. **Reportar Problemas**
   - Revisa si el problema ya ha sido reportado en la sección de [Issues](https://github.com/tu-usuario/med-storm/issues).
   - Si no existe, crea un nuevo issue con una descripción clara del problema o mejora propuesta.

2. **Desarrollo**
   - Haz un fork del repositorio.
   - Crea una rama descriptiva para tu característica o corrección: `git checkout -b feature/nueva-funcionalidad` o `fix/correcion-error`.
   - Realiza tus cambios siguiendo las guías de estilo del proyecto.
   - Asegúrate de que todas las pruebas pasen correctamente.
   - Envía un Pull Request (PR) con una descripción clara de los cambios realizados.

## Estándares de Código

- Sigue las convenciones de estilo PEP 8 para Python.
- Escribe pruebas unitarias para nuevo código.
- Documenta funciones y clases con docstrings siguiendo el formato Google Style.
- Mantén los commits atómicos y con mensajes descriptivos.

## Configuración del Entorno de Desarrollo

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/med-storm.git
   cd med-storm
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias de desarrollo:
   ```bash
   pip install -e .[dev]
   ```

4. Instala los pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Ejecutando Pruebas

```bash
pytest
```

## Enviando un Pull Request

1. Asegúrate de que todas las pruebas pasen.
2. Actualiza la documentación si es necesario.
3. Asegúrate de que tu código cumple con las guías de estilo.
4. Envía el PR a la rama `main`.

## Código de Conducta

Al participar en este proyecto, aceptas cumplir con nuestro [Código de Conducta](CODE_OF_CONDUCT.md).

## Agradecimientos

¡Gracias por contribuir a hacer de Med-STORM una herramienta mejor para todos!
