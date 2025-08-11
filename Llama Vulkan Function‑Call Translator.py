# -*- coding: utf-8 -*-
"""
Extension : Llama Vulkan Function‑Call Translator

Ce module intercepte les appels de fonction générés par le modèle,
les convertit en requêtes compatibles avec l’API de Vulkan Llama.cpp
et renvoie la réponse au chat.
"""

from .extensions import NewelleExtension
import json
import re
import subprocess  # pour appeler le binaire llama.cpp

# ------------------------------------------------------------------
class LlamaVulkanExtension(NewelleExtension):
    """
    Extension qui traduit les function‑calls du modèle en requêtes
    Vulkan Llama.cpp (LMStudio).
    """

    name = "Llama Vulkan Function‑Call Translator"
    id   = "llama_vulkan"

    # ------------------------------------------------------------------
    def install(self) -> None:
        """Pas besoin d’installer de dépendances supplémentaires."""
        pass

    # ------------------------------------------------------------------
    def preprocess_history(
            self,
            history: list[dict],
            prompts: list[dict]
        ) -> tuple[list, list]:
        """
        Avant l’appel au LLM : on ne fait rien.
        On laisse le texte tel quel pour que le modèle puisse générer
        un function‑call.
        """
        return history, prompts

    # ------------------------------------------------------------------
    def postprocess_history(
            self,
            history: list[dict],
            bot_response: str
        ) -> tuple[list, str]:
        """
        Après réception de la réponse du LLM.

        1. On cherche un bloc JSON qui représente un function‑call.
           Le format attendu est celui d’OpenAI :
               {"name":"search","arguments":{"q":"python"}}
        2. Si trouvé, on le transforme en requête Vulkan
           (exemple : `--function search --query "python"`).
        3. On exécute le binaire llama.cpp avec ces arguments.
        4. La sortie du binaire est insérée dans la réponse finale.

        Si aucun function‑call n’est détecté, on renvoie simplement
        la réponse brute.
        """
        # 1️⃣ Recherche d’un JSON valide dans bot_response
        func_call = self._extract_function_call(bot_response)
        if not func_call:
            return history, bot_response

        # 2️⃣ Traduction en arguments Vulkan
        vulkan_args = self._translate_to_vulkan(func_call)

        # 3️⃣ Exécution du binaire llama.cpp (exemple simplifié)
        result_text = self._run_llama_cpp(vulkan_args)

        # 4️⃣ Construction de la réponse finale
        final_response = (
            f"**Résultat de l’appel fonction‑call :**\n\n{result_text}"
        )
        return history, final_response

    # ------------------------------------------------------------------
    def _extract_function_call(self, text: str) -> dict | None:
        """
        Cherche un bloc JSON dans le texte. Retourne le dictionnaire
        ou None si rien trouvé.
        """
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
            # On vérifie qu’il contient bien 'name' et 'arguments'
            if "name" in obj and "arguments" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        return None

    # ------------------------------------------------------------------
    def _translate_to_vulkan(self, func: dict) -> list[str]:
        """
        Convertit le dictionnaire fonction‑call en liste d’arguments
        que llama.cpp accepte via la ligne de commande.
        Exemple :
          {"name":"search","arguments":{"q":"python"}}
          → ["--function", "search", "--query", "python"]
        """
        name = func["name"]
        args = func.get("arguments", {})

        # On suppose que chaque clé devient un flag `--<key>`
        vulkan_args = ["--function", name]
        for k, v in args.items():
            vulkan_args.extend([f"--{k}", str(v)])
        return vulkan_args

    # ------------------------------------------------------------------
    def _run_llama_cpp(self, args: list[str]) -> str:
        """
        Exécute le binaire llama.cpp avec les arguments fournis.
        Vous devez adapter le chemin vers votre exécutable et
        éventuellement ajouter d’autres options (model path, etc.).
        """
        # Chemin vers l’exécutable (à ajuster)
        exe_path = "/usr/local/bin/llama"  # ou "./llama"

        cmd = [exe_path] + args

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if completed.returncode != 0:
                return f"[Erreur] {completed.stderr.strip()}"
            return completed.stdout.strip()
        except Exception as e:
            return f"[Exception] {e}"

# ------------------------------------------------------------------
