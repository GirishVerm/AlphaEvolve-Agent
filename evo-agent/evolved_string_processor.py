#!/usr/bin/env python3
"""
Evolved String Processor
=======================

This is the evolved code from the interactive agent system.
The agent evolved this code through multiple iterations and improvements.

Usage: python3 evolved_string_processor.py
"""

class StringProcessorError(Exception):
    """Excepción genérica para errores en el procesador de cadenas."""
    pass


def string_processor(s: str, operation: str, **kwargs) -> str:
    """
    Procesa la cadena de entrada según la operación indicada.

    Args:
        s (str): Cadena de entrada a procesar.
        operation (str): Nombre de la operación. Opciones soportadas:
                         'upper', 'lower', 'capitalize', 'title',
                         'reverse', 'strip', 'replace'.
        **kwargs: Para la operación 'replace', debe incluir:
                  - old (str): subcadena a reemplazar.
                  - new (str): subcadena de reemplazo.

    Returns:
        str: Cadena resultante luego de aplicar la operación.

    Raises:
        StringProcessorError: Si falla la validación de entradas,
                              si la operación es desconocida o
                              si ocurre un error inesperado.
    """
    # Validación de tipo y contenido de la cadena de entrada.
    if not isinstance(s, str):
        raise StringProcessorError(
            f"El parámetro 's' debe ser str, no {type(s).__name__!r}"
        )
    # Validación de tipo de la operación.
    if not isinstance(operation, str):
        raise StringProcessorError(
            f"El parámetro 'operation' debe ser str, no {type(operation).__name__!r}"
        )

    # Mapeo de operaciones simples a funciones de string de Python.
    ops = {
        'upper': str.upper,
        'lower': str.lower,
        'capitalize': str.capitalize,
        'title': str.title,
        'reverse': lambda text: text[::-1],  # Reversión Unicode-safe
        'strip': str.strip,
    }

    try:
        if operation == 'replace':
            # Validar argumentos específicos de replace
            extra = set(kwargs) - {'old', 'new'}
            if extra:
                raise StringProcessorError(
                    f"Argumentos inesperados para 'replace': {sorted(extra)}"
                )
            old = kwargs.get('old')
            new = kwargs.get('new')
            # Verificar presencia y tipo de old/new
            if old is None or new is None:
                raise StringProcessorError(
                    "Faltan 'old' o 'new' para la operación 'replace'"
                )
            if not isinstance(old, str) or not isinstance(new, str):
                raise StringProcessorError(
                    "'old' y 'new' deben ser de tipo str"
                )
            # Realizar el reemplazo usando la API nativa de Python
            return s.replace(old, new)

        elif operation in ops:
            # No se esperan kwargs para las demás operaciones
            if kwargs:
                raise StringProcessorError(
                    f"No se esperan argumentos adicionales para la operación '{operation}'"
                )
            func = ops[operation]
            return func(s)

        else:
            # Operación no reconocida
            valid_ops = sorted(list(ops.keys()) + ['replace'])
            raise StringProcessorError(
                f"Operación desconocida '{operation}'. "
                f"Elija una de: {valid_ops}"
            )

    except StringProcessorError:
        # Propagar errores previstos sin modificación
        raise
    except Exception as e:
        # Capturar cualquier excepción inesperada y envolverla
        raise StringProcessorError(
            f"Error procesando '{operation}': {e}"
        ) from e


def run_demo():
    """Run a comprehensive demo of the evolved string processor."""
    print("🎯 EVOLVED STRING PROCESSOR DEMO")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        (" Héllo Wörld ", "strip", {}, "Strip whitespace"),
        (" Héllo Wörld ", "upper", {}, "Convert to uppercase"),
        ("héllo", "capitalize", {}, "Capitalize first letter"),
        ("hello world", "title", {}, "Title case"),
        ("hello", "lower", {}, "Convert to lowercase"),
        ("spa ce", "replace", {'old': " ", 'new': ""}, "Replace spaces"),
        ("abc😊def", "reverse", {}, "Reverse with emoji"),
        ("", "strip", {}, "Empty string"),
        ("   ", "strip", {}, "Only whitespace"),
    ]
    
    print(f"{'Operation':<12} {'Input':<20} {'Output':<20} {'Description'}")
    print("-" * 70)
    
    for input_str, operation, kwargs, description in test_cases:
        try:
            result = string_processor(input_str, operation, **kwargs)
            print(f"{operation:<12} {repr(input_str):<20} {repr(result):<20} {description}")
        except StringProcessorError as e:
            print(f"{operation:<12} {repr(input_str):<20} {'ERROR':<20} {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")


def interactive_mode():
    """Run the string processor in interactive mode."""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 30)
    print("Available operations: upper, lower, capitalize, title, reverse, strip, replace")
    print("For replace operation, you'll be asked for 'old' and 'new' values.")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            # Get input string
            input_str = input("\nEnter a string (or 'quit'): ").strip()
            if input_str.lower() == 'quit':
                break
            
            # Get operation
            operation = input("Enter operation: ").strip().lower()
            
            # Handle replace operation
            kwargs = {}
            if operation == 'replace':
                old = input("Enter 'old' string to replace: ").strip()
                new = input("Enter 'new' string: ").strip()
                kwargs = {'old': old, 'new': new}
            
            # Process the string
            result = string_processor(input_str, operation, **kwargs)
            print(f"✅ Result: {repr(result)}")
            
        except StringProcessorError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("🚀 EVOLVED STRING PROCESSOR")
    print("=" * 50)
    print("This code was evolved by an AI agent through multiple iterations!")
    print("=" * 50)
    
    # Run demo
    run_demo()
    
    # Ask if user wants interactive mode
    choice = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_mode()
    
    print("\n🎉 Thanks for trying the evolved string processor!") 