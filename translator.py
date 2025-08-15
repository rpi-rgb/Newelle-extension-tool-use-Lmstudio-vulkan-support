# -*- coding: utf-8 -*-
"""
Extension : Llama Vulkan Function‑Call Translator

Ce module intercepte les appels de fonction générés par le modèle,
les exécute via un processus llama.cpp persistant et renvoie la réponse au chat.
"""

from .extensions import NewelleExtension
import json
import re
import subprocess
import threading
import time
from queue import Queue, Empty

# ------------------------------------------------------------------
class LlamaVulkanExtension(NewelleExtension):
    """
    Extension qui traduit les function‑calls du modèle en requêtes
    Vulkan Llama.cpp (LMStudio).
    """

    name = "Llama Vulkan Function‑Call Translator (Stable)"
    id   = "llama_vulkan_stable"

    def __init__(self):
        super().__init__()
        self.llama_process = None
        self.output_queue = Queue()
        self.output_thread = None
        self.prompt = "> "
        self.executable_path = "/usr/local/bin/llama"  # IMPORTANT: Ajustez ce chemin
        self.model_path = "path/to/your/model.gguf" # IMPORTANT: Ajustez ce chemin

    # ------------------------------------------------------------------
    def install(self) -> None:
        """
        Démarre le processus llama.cpp en arrière-plan et attend qu'il soit prêt.
        """
        print("--- Llama Vulkan Function-Call Translator (Stable) ---")
        self.start_llama_process()
        print("-------------------------------------------------")

    def shutdown(self):
        """
        Arrête proprement le processus llama.cpp.
        """
        if self.llama_process and self.llama_process.poll() is None:
            print("Stopping llama.cpp process...")
            self.llama_process.terminate()
            try:
                self.llama_process.wait(timeout=5)
                print("Llama.cpp process stopped.")
            except subprocess.TimeoutExpired:
                print("Llama.cpp process did not terminate gracefully, killing.")
                self.llama_process.kill()
        self.llama_process = None

    def start_llama_process(self):
        """
        Lance le binaire llama.cpp en mode interactif et attend le prompt initial.
        """
        if self.llama_process and self.llama_process.poll() is None:
            print("Llama.cpp process is already running.")
            return

        cmd = [
            self.executable_path,
            "--model", self.model_path,
            "--interactive-first",
            "-i",
            # Ajoutez d'autres options CLI de llama.cpp si nécessaire
        ]

        try:
            print(f"Starting llama.cpp with command: {' '.join(cmd)}")
            self.llama_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirige stderr vers stdout
                text=True,
                bufsize=1,
                encoding='utf-8'
            )

            self.output_thread = threading.Thread(target=self._read_output)
            self.output_thread.daemon = True
            self.output_thread.start()

            print("Waiting for llama.cpp to be ready (this may take a while)...")
            self._wait_for_prompt(timeout=120) # Timeout généreux pour le chargement du modèle
            print("Llama.cpp is ready.")

        except FileNotFoundError:
            print(f"[ERREUR] Binaire llama.cpp non trouvé à '{self.executable_path}'.")
            print("Veuillez vérifier le chemin dans `translator.py`.")
            self.llama_process = None
        except Exception as e:
            print(f"[Exception] Impossible de démarrer llama.cpp : {e}")
            self.llama_process = None

    def _read_output(self):
        """
        Lit la sortie stdout du processus et la met dans une queue.
        """
        for line in iter(self.llama_process.stdout.readline, ''):
            self.output_queue.put(line)
        self.llama_process.stdout.close()

    def _wait_for_prompt(self, timeout: int) -> str:
        """
        Attend que le prompt ('> ') apparaisse dans la sortie, avec un timeout.
        Retourne tout le texte reçu avant le prompt.
        """
        buffer = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=1)
                if self.prompt in line:
                    # Nettoie la ligne du prompt et ajoute le reste au buffer
                    clean_line = line.split(self.prompt, 1)[0]
                    if clean_line:
                        buffer.append(clean_line)
                    return "".join(buffer)
                buffer.append(line)
            except Empty:
                if self.llama_process.poll() is not None:
                    raise RuntimeError("Le processus llama.cpp s'est terminé inopinément.")
                continue # Continue d'attendre
        raise TimeoutError("Timeout en attendant le prompt de llama.cpp.")

    # ------------------------------------------------------------------
    def preprocess_history(
            self,
            history: list[dict],
            prompts: list[dict]
        ) -> tuple[list, list]:
        return history, prompts

    # ------------------------------------------------------------------
    def postprocess_history(
            self,
            history: list[dict],
            bot_response: str
        ) -> tuple[list, str]:
        func_call = self._extract_function_call(bot_response)
        if not func_call:
            return history, bot_response

        vulkan_args = self._translate_to_vulkan(func_call)
        result_text = self._run_llama_command(vulkan_args)

        final_response = (
            f"**Résultat de l’appel de fonction :**\n\n```\n{result_text.strip()}\n```"
        )
        return history, final_response

    # ------------------------------------------------------------------
    def _extract_function_call(self, text: str) -> dict | None:
        """
        Cherche un bloc JSON valide dans le texte.
        """
        # Regex pour trouver un bloc JSON, même avec du texte autour
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            # Tente de parser le JSON le plus large possible
            obj = json.loads(match.group(0))
            if "name" in obj and "arguments" in obj:
                return obj
        except json.JSONDecodeError:
            # Fallback pour trouver un JSON valide à l'intérieur du texte
            for i in range(len(text)):
                if text[i] == '{':
                    try:
                        obj, _ = json.JSONDecoder().raw_decode(text[i:])
                        if "name" in obj and "arguments" in obj:
                            return obj
                    except json.JSONDecodeError:
                        continue
        return None

    # ------------------------------------------------------------------
    def _translate_to_vulkan(self, func: dict) -> list[str]:
        name = func["name"]
        args = func.get("arguments", {})
        vulkan_args = ["--function", name]
        for k, v in args.items():
            vulkan_args.extend([f"--{k}", str(v)])
        return vulkan_args

    # ------------------------------------------------------------------
    def _run_llama_command(self, args: list[str]) -> str:
        """
        Envoie une commande au processus llama.cpp persistant et attend la réponse.
        """
        if not self.llama_process or self.llama_process.poll() is not None:
            return "[ERREUR] Le processus Llama.cpp n'est pas actif. Essayez de redémarrer l'extension."

        command = " ".join(args) + "\n"

        # Vide la queue de toute sortie précédente avant d'envoyer la commande
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Empty:
                break

        self.llama_process.stdin.write(command)
        self.llama_process.stdin.flush()

        try:
            # Attend la réponse, qui se termine par le prompt
            response = self._wait_for_prompt(timeout=30)
            # La réponse est ce qui précède le prompt.
            # Il faut peut-être la nettoyer des échos de la commande.
            # Pour l'instant, on retourne la sortie brute.
            return response
        except TimeoutError:
            return "[ERREUR] Timeout en attendant la réponse de Llama.cpp."
        except RuntimeError as e:
            return f"[ERREUR] {e}"

# ------------------------------------------------------------------
# Assurez-vous d'appeler shutdown() lorsque Newelle se ferme.
# Par exemple, si Newelle a un mécanisme de "on_exit" ou "on_unload".
# Dans Newelle, cela pourrait être géré par le cycle de vie de l'extension.
# ------------------------------------------------------------------
