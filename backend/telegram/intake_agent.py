from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from backend.services.analysis_service import AnalysisService
from backend.services.claim_repository import ClaimRepository
from backend.telegram.repository import TelegramRepository


Parser = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class IntakeStep:
    key: str
    prompt: str
    parser: Parser
    explanation: str = ""
    buttons: Tuple[Tuple[str, ...], ...] = ()
    placeholder: Optional[str] = None


class IntakeValidationError(ValueError):
    pass


class TelegramIntakeAgent:
    """Conversational claim intake mapped to the exact deployed ML contract.

    The agent asks human-readable questions and converts the answers to the
    Kaggle feature representation expected by the fraud pipeline. Derived
    temporal fields and categorical buckets are created automatically.
    """

    def __init__(
        self,
        repository: Optional[TelegramRepository] = None,
        claim_repository: Optional[ClaimRepository] = None,
        analysis_service: Optional[AnalysisService] = None,
    ):
        self.repository = repository or TelegramRepository()
        self.claim_repository = claim_repository or ClaimRepository()
        self.analysis_service = analysis_service or AnalysisService()
        self.project_root = Path(__file__).resolve().parents[2]
        self._catalog = self._load_catalog()
        self.steps = self._build_steps()

    def _load_catalog(self) -> Dict[str, List[str]]:
        path = self.project_root / "data" / "kaggle" / "fraud_oracle.csv"
        df = pd.read_csv(path)
        catalog: Dict[str, List[str]] = {}
        for column in df.columns:
            dtype = df[column].dtype
            # Pandas 3.x puede inferir texto como StringDtype/str en lugar de
            # object. Comprobar solo ``dtype == object`` dejaba el catálogo
            # categórico vacío y hacía que valores válidos como Toyota/BMW
            # fueran rechazados por el bot.
            if (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                )
                catalog[column] = sorted(v for v in values.unique().tolist() if v)
        return catalog

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).casefold()

    @staticmethod
    def _int(text: str, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
        cleaned = re.sub(r"[^0-9-]", "", text.strip())
        if not cleaned or cleaned == "-":
            raise IntakeValidationError("Necesito un número entero.")
        value = int(cleaned)
        if value < minimum or (maximum is not None and value > maximum):
            range_text = f"entre {minimum} y {maximum}" if maximum is not None else f"mayor o igual que {minimum}"
            raise IntakeValidationError(f"El valor debe ser {range_text}.")
        return value

    @staticmethod
    def _date(text: str) -> datetime:
        value = text.strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise IntakeValidationError("Usa una fecha como 12/09/2026.")

    def _choice(self, feature: str, aliases: Optional[Dict[str, str]] = None) -> Parser:
        allowed = self._catalog.get(feature, [])
        lookup = {self._normalize(value): value for value in allowed}
        for key, value in (aliases or {}).items():
            lookup[self._normalize(key)] = value

        def parser(text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
            key = self._normalize(text)
            if key not in lookup:
                examples = ", ".join(allowed[:8])
                raise IntakeValidationError(f"Valor no reconocido. Ejemplos válidos: {examples}.")
            return {feature: lookup[key]}

        return parser

    @staticmethod
    def _month_abbr(date: datetime) -> str:
        return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][date.month - 1]

    @staticmethod
    def _weekday(date: datetime) -> str:
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][date.weekday()]

    @staticmethod
    def _week_of_month(date: datetime) -> int:
        return min(5, ((date.day - 1) // 7) + 1)

    def _accident_date(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        date = self._date(text)
        return {
            "Month": self._month_abbr(date),
            "WeekOfMonth": self._week_of_month(date),
            "DayOfWeek": self._weekday(date),
            "Year": date.year,
        }

    def _claim_date(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        date = self._date(text)
        return {
            "MonthClaimed": self._month_abbr(date),
            "WeekOfMonthClaimed": self._week_of_month(date),
            "DayOfWeekClaimed": self._weekday(date),
        }

    def _vehicle_price(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        if value < 20000:
            bucket = "less than 20000"
        elif value < 30000:
            bucket = "20000 to 29000"
        elif value < 40000:
            bucket = "30000 to 39000"
        elif value < 60000:
            bucket = "40000 to 59000"
        elif value < 70000:
            bucket = "60000 to 69000"
        else:
            bucket = "more than 69000"
        return {"VehiclePrice": bucket}

    def _days_policy_accident(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        if value == 0:
            bucket = "none"
        elif value <= 7:
            bucket = "1 to 7"
        elif value <= 15:
            bucket = "8 to 15"
        elif value <= 30:
            bucket = "15 to 30"
        else:
            bucket = "more than 30"
        return {"Days_Policy_Accident": bucket}

    def _days_policy_claim(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        if value == 0:
            bucket = "none"
        elif value <= 15:
            bucket = "8 to 15"
        elif value <= 30:
            bucket = "15 to 30"
        else:
            bucket = "more than 30"
        return {"Days_Policy_Claim": bucket}

    def _past_claims(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        bucket = "none" if value == 0 else "1" if value == 1 else "2 to 4" if value <= 4 else "more than 4"
        return {"PastNumberOfClaims": bucket}

    def _vehicle_age(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        if value <= 1:
            bucket = "new"
        elif value <= 7:
            bucket = f"{value} years"
        else:
            bucket = "more than 7"
        if bucket not in self._catalog["AgeOfVehicle"]:
            # The source dataset starts at 2 years; 0/1 is represented as new.
            bucket = "new"
        return {"AgeOfVehicle": bucket}

    def _holder_age(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=16, maximum=120)
        if value <= 17:
            bucket = "16 to 17"
        elif value <= 20:
            bucket = "18 to 20"
        elif value <= 25:
            bucket = "21 to 25"
        elif value <= 30:
            bucket = "26 to 30"
        elif value <= 35:
            bucket = "31 to 35"
        elif value <= 40:
            bucket = "36 to 40"
        elif value <= 50:
            bucket = "41 to 50"
        elif value <= 65:
            bucket = "51 to 65"
        else:
            bucket = "over 65"
        return {"AgeOfPolicyHolder": bucket}

    def _supplements(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=0)
        bucket = "none" if value == 0 else "1 to 2" if value <= 2 else "3 to 5" if value <= 5 else "more than 5"
        return {"NumberOfSuppliments": bucket}

    def _address_change(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        months = self._int(text, minimum=0)
        if months == 0:
            bucket = "no change"
        elif months < 6:
            bucket = "under 6 months"
        elif months <= 18:
            bucket = "1 year"
        elif months <= 42:
            bucket = "2 to 3 years"
        else:
            bucket = "4 to 8 years"
        return {"AddressChange_Claim": bucket}

    def _number_of_cars(self, text: str, _features: Dict[str, Any]) -> Dict[str, Any]:
        value = self._int(text, minimum=1)
        bucket = "1 vehicle" if value == 1 else "2 vehicles" if value == 2 else "3 to 4" if value <= 4 else "5 to 8" if value <= 8 else "more than 8"
        return {"NumberOfCars": bucket}

    def _base_policy(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            "todo riesgo": "All Perils",
            "all perils": "All Perils",
            "colision": "Collision",
            "colisión": "Collision",
            "collision": "Collision",
            "responsabilidad civil": "Liability",
            "liability": "Liability",
        }
        result = self._choice("BasePolicy", aliases)(text, features)
        category = features.get("VehicleCategory")
        if category:
            result["PolicyType"] = f"{category} - {result['BasePolicy']}"
        return result

    def _yes_no(self, feature: str) -> Parser:
        return self._choice(feature, {"si": "Yes", "sí": "Yes", "no": "No", "yes": "Yes"})

    def _build_steps(self) -> List[IntakeStep]:
        return [
            IntakeStep(
                "accident_date",
                "1/24 · 📅 Fecha del accidente",
                self._accident_date,
                "La fecha permite derivar automáticamente el mes, el día de la semana, la semana del mes y el año que usa el modelo.",
                placeholder="Ej.: 12/09/2026",
            ),
            IntakeStep(
                "claim_date",
                "2/24 · 📨 Fecha de comunicación del siniestro",
                self._claim_date,
                "Indica cuándo se notificó el siniestro. A partir de esta fecha se generan las variables temporales de comunicación.",
                placeholder="Ej.: 13/09/2026",
            ),
            IntakeStep(
                "Make",
                "3/24 · 🚗 Marca del vehículo",
                self._choice(
                    "Make",
                    {
                        "Volkswagen": "VW",
                        "Acura": "Accura",
                        "Nissan": "Nisson",
                        "Porsche": "Porche",
                        "Mercedes": "Mecedes",
                        "Mercedes-Benz": "Mecedes",
                    },
                ),
                "Es una variable categórica del modelo. Puedes tocar una marca frecuente o escribir manualmente cualquier marca válida del catálogo.",
                (("Toyota", "Ford", "BMW"), ("Honda", "VW", "Mazda")),
                "Escribe una marca, por ejemplo Toyota",
            ),
            IntakeStep(
                "AccidentArea",
                "4/24 · 🏙️ Zona del accidente",
                self._choice("AccidentArea", {"urbana": "Urban", "urbano": "Urban", "rural": "Rural"}),
                "Clasifica el entorno en el que ocurrió el accidente. Solo necesitamos distinguir entre zona urbana y rural.",
                (("Urbana", "Rural"),),
            ),
            IntakeStep(
                "Sex",
                "5/24 · 👤 Sexo del conductor",
                self._choice("Sex", {"hombre": "Male", "varón": "Male", "mujer": "Female"}),
                "Este campo forma parte del dataset de referencia y se conserva para reproducir exactamente el contrato del modelo entrenado.",
                (("Hombre", "Mujer"),),
            ),
            IntakeStep(
                "MaritalStatus",
                "6/24 · 💍 Estado civil",
                self._choice("MaritalStatus", {"soltero": "Single", "soltera": "Single", "casado": "Married", "casada": "Married", "divorciado": "Divorced", "divorciada": "Divorced", "viudo": "Widow", "viuda": "Widow"}),
                "Selecciona la categoría que corresponde al conductor. Se normaliza a las etiquetas del dataset antes de analizar el claim.",
                (("Soltero", "Casado"), ("Divorciado", "Viudo")),
            ),
            IntakeStep(
                "Age",
                "7/24 · 🎂 Edad del conductor",
                lambda text, _: {"Age": self._int(text, minimum=0, maximum=120)},
                "Introduce la edad en años. El sistema valida automáticamente que sea un valor numérico coherente.",
                placeholder="Ej.: 28",
            ),
            IntakeStep(
                "Fault",
                "8/24 · ⚖️ Responsabilidad del accidente",
                self._choice("Fault", {"asegurado": "Policy Holder", "tomador": "Policy Holder", "tercero": "Third Party"}),
                "Indica a quién se atribuye la responsabilidad principal según la información disponible del siniestro.",
                (("Asegurado", "Tercero"),),
            ),
            IntakeStep(
                "VehicleCategory",
                "9/24 · 🚘 Categoría del vehículo",
                self._choice("VehicleCategory", {"sedán": "Sedan", "sedan": "Sedan", "deportivo": "Sport", "sport": "Sport", "utilitario": "Utility", "utility": "Utility"}),
                "Selecciona la categoría usada por el dataset. Esta elección se combina internamente con la modalidad del seguro para construir una variable técnica del modelo.",
                (("Sedan", "Sport", "Utility"),),
            ),
            IntakeStep(
                "BasePolicy",
                "10/24 · 🛡️ Modalidad del seguro",
                self._base_policy,
                "Selecciona la modalidad contratada. Este dato estructurado forma parte del modelo de fraude; se utiliza únicamente como dato estructurado del modelo.",
                (("Todo riesgo", "Colisión"), ("Responsabilidad civil",)),
            ),
            IntakeStep(
                "VehiclePrice",
                "11/24 · 💶 Valor aproximado del vehículo",
                self._vehicle_price,
                "Introduce un importe numérico en euros. El sistema lo transforma automáticamente al tramo de precio utilizado por el modelo.",
                placeholder="Ej.: 32000",
            ),
            IntakeStep(
                "Deductible",
                "12/24 · 💳 Franquicia / deducible",
                lambda text, _: {"Deductible": self._int(text, minimum=0)},
                "Indica el importe de la franquicia del seguro. Escribe solo el número; por ejemplo, 400.",
                placeholder="Ej.: 400",
            ),
            IntakeStep(
                "DriverRating",
                "13/24 · ⭐ Rating del conductor",
                lambda text, _: {"DriverRating": self._int(text, minimum=1, maximum=4)},
                "El dataset utiliza una escala de 1 a 4. Toca uno de los botones para evitar errores de formato.",
                (("1", "2", "3", "4"),),
            ),
            IntakeStep(
                "Days_Policy_Accident",
                "14/24 · ⏱️ Antigüedad del seguro al ocurrir el accidente",
                self._days_policy_accident,
                "Introduce cuántos días llevaba activo el seguro cuando ocurrió el accidente. InsurGuard agrupa después el valor en el tramo correspondiente.",
                (("0", "7", "15", "30"),),
                "Número de días, por ejemplo 120",
            ),
            IntakeStep(
                "Days_Policy_Claim",
                "15/24 · 🗓️ Antigüedad del seguro al comunicar el siniestro",
                self._days_policy_claim,
                "Indica los días de antigüedad del seguro cuando se notificó el siniestro. El valor se convierte automáticamente a la categoría del dataset.",
                (("0", "15", "30", "60"),),
                "Número de días, por ejemplo 121",
            ),
            IntakeStep(
                "PastNumberOfClaims",
                "16/24 · 📚 Reclamaciones anteriores",
                self._past_claims,
                "Indica cuántas reclamaciones anteriores constan para el asegurado. El sistema las agrupa en los intervalos utilizados por el modelo.",
                (("0", "1", "2", "3", "5"),),
            ),
            IntakeStep(
                "AgeOfVehicle",
                "17/24 · 🚙 Antigüedad del vehículo",
                self._vehicle_age,
                "Escribe la antigüedad en años. Usa 0 si el vehículo es nuevo; InsurGuard normaliza el dato a la categoría del dataset.",
                (("0", "2", "3", "5", "8"),),
            ),
            IntakeStep(
                "AgeOfPolicyHolder",
                "18/24 · 👤 Edad del titular del seguro",
                self._holder_age,
                "Introduce la edad del titular, no necesariamente la del conductor. El sistema la transforma al tramo de edad correspondiente.",
                placeholder="Ej.: 30",
            ),
            IntakeStep(
                "PoliceReportFiled",
                "19/24 · 👮 Parte o atestado policial",
                self._yes_no("PoliceReportFiled"),
                "Indica si existe un parte o atestado policial asociado al accidente.",
                (("Sí", "No"),),
            ),
            IntakeStep(
                "WitnessPresent",
                "20/24 · 👁️ Testigos",
                self._yes_no("WitnessPresent"),
                "Indica si hubo al menos un testigo del accidente según la información disponible.",
                (("Sí", "No"),),
            ),
            IntakeStep(
                "AgentType",
                "21/24 · 🧑‍💼 Tipo de agente",
                self._choice("AgentType", {"interno": "Internal", "externo": "External"}),
                "Selecciona si el expediente fue gestionado por un agente interno o externo.",
                (("Interno", "Externo"),),
            ),
            IntakeStep(
                "NumberOfSuppliments",
                "22/24 · 📎 Suplementos del expediente",
                self._supplements,
                "Indica cuántos suplementos o ampliaciones documentales están asociados a la reclamación.",
                (("0", "1", "2", "3", "6"),),
            ),
            IntakeStep(
                "AddressChange_Claim",
                "23/24 · 🏠 Cambio de dirección",
                self._address_change,
                "Indica cuántos meses han pasado desde el último cambio de dirección. Usa 0 si no hubo cambio registrado.",
                (("0", "3", "12", "24", "48"),),
                "Meses desde el último cambio",
            ),
            IntakeStep(
                "NumberOfCars",
                "24/24 · 🚗 Vehículos asociados al asegurado",
                self._number_of_cars,
                "Indica cuántos vehículos están asociados al asegurado. El valor se agrupa automáticamente en la categoría del dataset.",
                (("1", "2", "3", "4", "5"),),
            ),
        ]

    @staticmethod
    def _new_claim_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"TG-{stamp}-{uuid.uuid4().hex[:4].upper()}"

    @staticmethod
    def mask_chat_id(chat_id: str) -> str:
        value = str(chat_id)
        if len(value) <= 4:
            return "*" * len(value)
        return f"{'*' * max(0, len(value) - 4)}{value[-4:]}"

    @staticmethod
    def _format_step(step: IntakeStep) -> str:
        parts = [step.prompt]
        if step.explanation:
            parts.append(f"ℹ️ {step.explanation}")
        if step.buttons:
            parts.append("👇 Puedes tocar una opción o escribir tu respuesta manualmente.")
        return "\n\n".join(parts)

    def _next_prompt(self, session: Dict[str, Any]) -> str:
        step = int(session.get("current_step", 0))
        if step >= len(self.steps):
            return "✅ Datos completos. Puedes adjuntar evidencia opcional o pulsar 🔎 Analizar."
        return self._format_step(self.steps[step])

    @staticmethod
    def _inline_keyboard(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        """Build an InlineKeyboardMarkup so buttons are always attached to the bot message."""
        return {
            "inline_keyboard": [
                [{"text": label, "callback_data": callback[:64]} for label, callback in row]
                for row in rows
                if row
            ]
        }

    def reply_markup_for(self, chat_id: str) -> Dict[str, Any]:
        """Return a context-aware inline keyboard for the current state.

        Inline buttons are attached directly below the message and cannot be hidden
        behind Telegram's custom-keyboard toggle. Free-text answers continue to work.
        """
        session = self.repository.get_session(chat_id)
        if not session or session.get("status") in {"cancelled", "completed"}:
            return self._inline_keyboard([
                [("🆕 Nuevo siniestro", "cmd:NUEVO"), ("📊 Estado", "cmd:ESTADO")],
                [("ℹ️ Ayuda", "cmd:AYUDA")],
            ])

        current_step = int(session.get("current_step", 0))
        if current_step >= len(self.steps):
            return self._inline_keyboard([
                [("🔎 Analizar", "cmd:ANALIZAR"), ("📊 Estado", "cmd:ESTADO")],
                [("❌ Cancelar", "cmd:CANCELAR"), ("ℹ️ Ayuda", "cmd:AYUDA")],
            ])

        step = self.steps[current_step]
        rows: List[List[Tuple[str, str]]] = [
            [(label, f"answer:{label}") for label in row]
            for row in step.buttons
        ]
        rows.append([("📊 Estado", "cmd:ESTADO"), ("❌ Cancelar", "cmd:CANCELAR")])
        return self._inline_keyboard(rows)

    def help_text(self) -> str:
        return (
            "🤖 InsurGuard AI · Asistente de siniestros\n\n"
            "🆕 NUEVO - iniciar un claim\n"
            "📊 ESTADO - ver progreso\n"
            "🔎 ANALIZAR - ejecutar el preanálisis de fraude\n"
            "❌ CANCELAR - cancelar la captura\n\n"
            "Durante el formulario aparecerán botones contextuales para las respuestas cerradas. "
            "También puedes adjuntar PDFs o imágenes como evidencia. Los adjuntos se conservan "
            "para que el supervisor pueda abrirlos y revisarlos en el expediente."
        )

    def start(self, chat_id: str) -> Tuple[Dict[str, Any], List[str]]:
        claim_id = self._new_claim_id()
        session = self.repository.reset_session(chat_id, claim_id)
        return session, [
            f"🆕 Nuevo siniestro creado: {claim_id}.",
            "Voy a recopilar los datos necesarios para el modelo. Las preguntas incluyen una breve explicación y, cuando sea posible, botones para responder más rápido.",
            self._next_prompt(session),
        ]

    def _status_text(self, session: Optional[Dict[str, Any]]) -> str:
        if not session:
            return "No hay una sesión activa. Escribe NUEVO para iniciar un siniestro."
        step = int(session.get("current_step", 0))
        completed = min(step, len(self.steps))
        blocks = round((completed / len(self.steps)) * 10) if self.steps else 0
        progress = "█" * blocks + "░" * (10 - blocks)
        return (
            f"📊 Estado del claim {session.get('claim_id')}\n"
            f"Estado: {session.get('status')}\n"
            f"Progreso: {progress} {completed}/{len(self.steps)}\n"
            f"Documentos adjuntos: {len(session.get('data', {}).get('documents', []))}\n\n"
            f"Siguiente paso:\n{self._next_prompt(session)}"
        )

    def _persist_claim(self, session: Dict[str, Any]) -> Dict[str, Any]:
        features = dict(session.get("data", {}).get("features", {}))
        claim = {
            "claim_id": session["claim_id"],
            "source": {
                "type": "telegram",
                "chat_id_masked": self.mask_chat_id(session["chat_id"]),
                "telegram_username": session.get("username"),
            },
            "features": features,
            "ground_truth": {},
            "documents": list(session.get("data", {}).get("documents", [])),
        }
        return self.claim_repository.save_claim(claim)

    def analyze_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        if int(session.get("current_step", 0)) < len(self.steps):
            raise IntakeValidationError("Aún faltan datos. " + self._next_prompt(session))
        claim = self._persist_claim(session)
        result = self.analysis_service.analyze(
            {
                "claim_id": claim["claim_id"],
                "features": claim["features"],
                "documents": claim.get("documents", []),
            },
            source_channel="telegram",
            record_production=True,
        )
        session["status"] = "completed"
        session["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.repository.save_session(session)
        return result

    @staticmethod
    def _analysis_reply(result: Dict[str, Any]) -> str:
        score = result.get("fraud_score")
        score_text = "N/D" if score is None else f"{float(score) * 100:.1f}%"
        return (
            "ANÁLISIS COMPLETADO\n"
            f"Claim: {result.get('claim_id')}\n"
            f"Probabilidad de fraude: {score_text}\n"
            f"Riesgo: {result.get('risk_level') or 'N/D'}\n"
            f"Revisión manual: {'Sí' if result.get('manual_review_required') else 'No'}\n"
            "El detalle completo del expediente queda disponible para el supervisor."
        )

    def process_text(self, chat_id: str, text: str) -> List[str]:
        value = (text or "").strip()
        button_commands = {
            "🆕 Nuevo siniestro": "NUEVO",
            "📊 Estado": "ESTADO",
            "🔎 Analizar": "ANALIZAR",
            "❌ Cancelar": "CANCELAR",
            "ℹ️ Ayuda": "AYUDA",
        }
        if value in button_commands:
            value = button_commands[value]
        # Telegram commands may arrive as /start or /start@BotName.
        command = value.split(maxsplit=1)[0].split("@", 1)[0].casefold() if value.startswith("/") else ""
        command_map = {
            "/nuevo": "NUEVO",
            "/estado": "ESTADO",
            "/analizar": "ANALIZAR",
            "/cancelar": "CANCELAR",
            "/ayuda": "AYUDA",
            "/help": "AYUDA",
        }
        if command in command_map:
            value = command_map[command]
        upper = value.upper()
        session = self.repository.get_session(chat_id)

        if command == "/start":
            return ["👋 Bienvenido a InsurGuard AI.", self.help_text()]
        if upper in {"AYUDA", "HELP", "MENU", "MENÚ"}:
            return [self.help_text()]
        if upper in {"NUEVO", "NUEVA", "NUEVO SINIESTRO", "NUEVO CLAIM"}:
            _, replies = self.start(chat_id)
            return replies
        if upper == "ESTADO":
            return [self._status_text(session)]
        if upper == "CANCELAR":
            if not session:
                return ["No hay una sesión activa."]
            session["status"] = "cancelled"
            self.repository.save_session(session)
            return [f"❌ Claim {session.get('claim_id')} cancelado. Pulsa 🆕 Nuevo siniestro cuando quieras empezar otro."]

        if not session or session.get("status") == "cancelled":
            return ["No hay un claim activo. Escribe NUEVO para comenzar.\n\n" + self.help_text()]

        if upper == "ANALIZAR":
            try:
                result = self.analyze_session(session)
                return [self._analysis_reply(result)]
            except IntakeValidationError as exc:
                return [str(exc)]

        if session.get("status") == "completed":
            return ["Este claim ya está completado. Escribe NUEVO para iniciar otro o ESTADO para verlo."]

        current_step = int(session.get("current_step", 0))
        step = self.steps[current_step]
        features = dict(session.get("data", {}).get("features", {}))
        try:
            updates = step.parser(value, features)
        except IntakeValidationError as exc:
            return [f"⚠️ No puedo guardar ese valor: {exc}\n\n{self._format_step(step)}"]
        features.update(updates)
        session.setdefault("data", {})["features"] = features
        session["current_step"] = current_step + 1
        self.repository.save_session(session)

        if session["current_step"] >= len(self.steps):
            self._persist_claim(session)
            return [
                "✅ Datos estructurados completos y claim guardado.",
                "Puedes adjuntar PDFs o imágenes como evidencia opcional. Se guardarán para que el supervisor pueda abrirlos y revisarlos en el expediente.",
                "Pulsa 🔎 Analizar cuando quieras ejecutar la validación y el modelo de fraude.",
            ]
        return ["✅ Dato guardado.", self._next_prompt(session)]

    def register_document(self, chat_id: str, file_path: Path, original_name: str) -> List[str]:
        session = self.repository.get_session(chat_id)
        if not session or session.get("status") == "cancelled":
            return ["Antes de adjuntar documentos, escribe NUEVO para crear un claim."]
        claim_id = session["claim_id"]
        safe_name = Path(original_name).name
        documents_dir = self.claim_repository.documents_dir(claim_id)
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / safe_name
        destination.write_bytes(file_path.read_bytes())
        documents = list(session.setdefault("data", {}).get("documents", []))
        if safe_name not in documents:
            documents.append(safe_name)
        session["data"]["documents"] = documents
        self.repository.save_session(session)
        if int(session.get("current_step", 0)) >= len(self.steps):
            self._persist_claim(session)
        return [f"📎 Documento {safe_name} asociado a {claim_id}. Total: {len(documents)}. Queda disponible para revisión del supervisor."]
