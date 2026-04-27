import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import graph

load_dotenv()


def chat():
    print("=" * 50)
    print("Agente Conversacional - Reto ICESI")
    print("Escribe 'salir' para terminar.")
    print("=" * 50)

    history = []

    while True:
        user_input = input("\nTu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            print("Sesion terminada.")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": history})
        history = result["messages"]

        last = result["messages"][-1]
        print(f"\nAgente: {last.content}")


if __name__ == "__main__":
    chat()