import streamlit as st
import random
from pymongo import MongoClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from bson.objectid import ObjectId
from datetime import datetime
from dotenv import load_dotenv
import os
import google.generativeai as genai
import html

load_dotenv()

# --- CONFIG & STYLING ---

st.set_page_config(layout="wide", page_title="Valutazione Modelli GreenBoom")

# --- FILTER CONFIGURATION ---
# Lista dei model_id da mostrare. Se vuota, mostra tutti i modelli.
# Esempio: ["greenboom"] per mostrare solo le risposte del modello greenboom
MODEL_FILTER = ["greenboom"]  # Lasciare vuoto [] per mostrare tutti i modelli

# --- TEXTS IN ITALIAN ---

TEXTS = {
    "login_header": "Accesso Valutatore V2",
    "username_label": "Nome Utente",
    "password_label": "Password",
    "login_button": "Accedi",
    "login_error": "Nome utente o password non validi",
    "welcome": "Benvenuto/a",
    "logout_button": "Esci",
    "progress_bar_text": "Progresso Totale Valutazioni",
    "all_eval_complete": "Congratulazioni! Hai completato tutte le valutazioni.",
    "session_eval_complete": "Hai valutato tutti i test per questa sessione.",
    "item_to_eval_header": "Test da Valutare",
    "question_header": "💬 Domanda",
    "model_answer_header": "😃 Risposta del Modello",
    "ground_truth_expander": "🟢 Mostra/Nascondi Ground Truth (Riferimento)",
    "your_eval_header": "La Tua Valutazione",
    
    "accuracy_label": "Accuratezza semantica",
    "completezza_label": "Completezza",
    "comprensibilità_label": "Comprensibilità",
    "precisione_label": "Precisione",
    "gradevolezza_label": "Gradevolezza",
    "allucinazione_label": "Rilevata Allucinazione",
    "esperto_label": "Sono esperto",
    
    "comment_label": "Commento/Motivazione",
    "submit_button": "✅ Invia e Prosegui",
    "submitted_info": "Valutazione già inviata. Procedi al prossimo.",
    "eval_submitted_success": "Valutazione inviata con successo!",
    "back_button": "⬅️ Indietro",
    "next_button": "Avanti ➡️",
    "title": "Piattaforma di Valutazione Umana per Baseline Models"
}

# --- DATABASE & DATA MODELS ---

credentials = {
    "Sara": os.getenv("JUDGE1_PASSWORD"),
    "Federico": os.getenv("JUDGE2_PASSWORD"),
    "Giorgia": os.getenv("JUDGE3_PASSWORD"),
    "Debora": os.getenv("JUDGE4_PASSWORD"),
    "Gianni": os.getenv("JUDGE5_PASSWORD"),
    "Marco": os.getenv("JUDGE6_PASSWORD"),
}

class Evaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    judge_id: str
    
    accuracy: int
    completezza: int
    comprensibilità: int
    precisione: int
    gradevolezza: int
    allucinazione: bool
    esperto: bool
    
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_id: str
    answer: str
    evaluations: List[Evaluation] = []
    created_at: Optional[datetime] = None

class Question(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: ObjectId = Field(default_factory=ObjectId, alias='_id')
    id_number: int
    created_at: Optional[datetime] = None
    text: str
    ground_truth: str
    model_answers: List[ModelAnswer] = []

@st.cache_resource
def get_db_connection():
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client['GreenBoom']

db = get_db_connection()
questions_collection = db['TestSet']

# --- APP LOGIC ---

def login_page():
    st.header(TEXTS["login_header"])
    username = st.text_input(TEXTS["username_label"], key="login_username")
    password = st.text_input(TEXTS["password_label"], type="password", key="login_password")

    if st.button(TEXTS["login_button"]):
        if username in credentials and credentials[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['judge_id'] = username
            st.rerun()
        else:
            st.error(TEXTS["login_error"])

def evaluation_page():
    st.sidebar.title(f"{TEXTS['welcome']}, {st.session_state['judge_id']}")
    if st.sidebar.button(TEXTS["logout_button"]):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.sidebar.markdown(f"""
                        👩🏻‍🏫 Valuta sempre la risposta sulla base del ground truth 👨🏻‍🏫 \n\n
                        **Accuratezza**: Quanto la risposta del modello é somigliante al ground truth? \n\n
                        **Completezza**: Quanto la risposta del modello copre tutti gli aspetti del ground truth? \n\n
                        **Comprensibilità**: Quanto é chiara e comprensibile la risposta del modello? \n\n
                        **Precisione**: Quanto la risposta riporta correttamente la fonte/normativa? \n\n
                        **Gradevolezza**: Quanto personalmente ti piace la risposta del modello? \n\n
                        **Allucinazione**: Ci sono informazioni totalmente inventate o false? (allucinazione) \n\n
                        **Sono esperto**: Se la valutazione é espressa dalla persona solitamente operante in materia. \n\n
                        **Commento**: Se trovi allucinazioni o fornisci valutazioni particolarmente positive o negative, spiega il perché. \n\n
                        """)

    @st.cache_data(show_spinner="Caricamento test di valutazione...")
    def _get_evaluation_queue_and_total_items(judge_id, model_filter):
        all_model_answers_flat = []
        evaluation_queue = []
        all_questions = list(questions_collection.find({}, {"_id": 1, "model_answers": 1}))
        
        for q in all_questions:
            for i, ma in enumerate(q.get('model_answers', [])):
                # Applica il filtro sui model_id se configurato
                if model_filter and ma.get('model_id') not in model_filter:
                    continue
                
                item = {'q_id': q['_id'], 'ma_idx': i}
                all_model_answers_flat.append(item)
                
                evaluators = [e['judge_id'] for e in ma.get('evaluations', [])]
                if judge_id not in evaluators:
                    evaluation_queue.append(item)
        
        random.shuffle(evaluation_queue)
        return evaluation_queue, len(all_model_answers_flat)

    if 'evaluation_queue' not in st.session_state:
        st.session_state.evaluation_queue, st.session_state.total_items = _get_evaluation_queue_and_total_items(st.session_state['judge_id'], tuple(MODEL_FILTER))
        st.session_state.evaluated_count = st.session_state.total_items - len(st.session_state.get('evaluation_queue', []))
        st.session_state.current_index = 0
    
    # Progress Bar
    progress = st.session_state.evaluated_count / st.session_state.total_items if st.session_state.total_items > 0 else 0
    st.progress(progress, text=f"{TEXTS['progress_bar_text']}: {st.session_state.evaluated_count}/{st.session_state.total_items}")

    queue = st.session_state.get('evaluation_queue', [])
    if not queue or st.session_state.current_index >= len(queue):
        st.success(TEXTS["all_eval_complete"])
        st.balloons()
        return

    index = st.session_state.current_index
    item = queue[index]
    question_doc = questions_collection.find_one({'_id': item['q_id']})
    model_answer = question_doc['model_answers'][item['ma_idx']]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(TEXTS["item_to_eval_header"])
        
        with st.container(border=True):
            st.markdown(f"**{TEXTS['question_header']}:**")
            st.markdown(f"> {html.unescape(question_doc['text'])}")
            
            st.markdown(f"**{TEXTS['model_answer_header']}:**")
            
            unescaped_answer = html.unescape(model_answer['answer'])
            blocked_answer = unescaped_answer #"\n".join([f"> {line}" for line in unescaped_answer.split('\n')])
            st.markdown(blocked_answer)
    
            with st.expander(TEXTS["ground_truth_expander"]):
                st.info(f"**Ground Truth:**\n\n{question_doc['ground_truth']}")

    with col2:
        st.subheader(TEXTS["your_eval_header"])
        with st.form(key=f"eval_form_{item['q_id']}_{item['ma_idx']}"):
            
            accuracy = st.slider(TEXTS["accuracy_label"], 0, 10, 5)
            completezza = st.slider(TEXTS["completezza_label"], 0, 10, 5)
            comprensibilità = st.slider(TEXTS["comprensibilità_label"], 0, 10, 5)
            precisione = st.slider(TEXTS["precisione_label"], 0, 10, 5)
            gradevolezza = st.slider(TEXTS["gradevolezza_label"], 0, 10, 5)
            allucinazione = st.checkbox(TEXTS["allucinazione_label"])
            esperto = st.checkbox(TEXTS["esperto_label"])
            comment = st.text_area(TEXTS["comment_label"])
            submit_button = st.form_submit_button(TEXTS["submit_button"])

            if submit_button:
                evaluation = Evaluation(
                    judge_id=st.session_state['judge_id'],
                    accuracy=accuracy, 
                    completezza=completezza,
                    comprensibilità=comprensibilità,
                    precisione=precisione,
                    gradevolezza=gradevolezza,
                    allucinazione=allucinazione,
                    esperto=esperto,
                    comment=comment
                )
                questions_collection.update_one(
                    {'_id': item['q_id']},
                    {'$push': {f'model_answers.{item["ma_idx"]}.evaluations': evaluation.model_dump()}}
                )
                st.session_state.evaluated_count += 1
                
                # Remove from queue and move to next
                st.session_state.evaluation_queue.pop(index)
                
                # Ensure index is not out of bounds for the next item
                if st.session_state.current_index >= len(st.session_state.evaluation_queue):
                    st.session_state.current_index = 0
                
                st.toast(TEXTS["eval_submitted_success"])
                st.rerun()

    # Navigation (outside columns, but can be styled)
    nav_cols = st.columns([1, 8, 1])
    with nav_cols[0]:
        if st.button(TEXTS["back_button"], use_container_width=True, disabled=(index <= 0)):
            st.session_state.current_index -= 1
            st.rerun()
    with nav_cols[2]:
        if st.button(TEXTS["next_button"], use_container_width=True, disabled=(index >= len(queue) - 1)):
            st.session_state.current_index += 1
            st.rerun()

if __name__ == "__main__":
    if st.session_state.get('logged_in', False):
        evaluation_page()
    else:
        login_page() 