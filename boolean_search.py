from __future__ import annotations

import argparse
import html
import inspect
import re
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any


# регулярные выражения для очистки HTML
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?>.*?</\1>", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

# термы в документах: кириллица, допускается дефис
TERM_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)

# токены булевого запроса
QUERY_TOKEN_RE = re.compile(
    r"\(|\)|AND|OR|NOT|[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)?",
    re.IGNORECASE,
)

OPERATORS = {"AND", "OR", "NOT"}
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
RIGHT_ASSOC = {"NOT"}


def patch_inspect_getargspec() -> None:
    if hasattr(inspect, "getargspec"):
        return

    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec


def find_and_attach_local_pymorphy2_paths(project_root: Path) -> None:
    """ищет соседние venv и добавляет site-packages с pymorphy2 в sys.path"""
    parent = project_root.parent
    for site_packages in parent.glob("*/.venv/lib/python*/site-packages"):
        if (site_packages / "pymorphy2").exists():
            sys.path.append(str(site_packages))


def get_morph_analyzer(project_root: Path) -> Any | None:
    """возвращает MorphAnalyzer (pymorphy3/pymorphy2) или None"""
    patch_inspect_getargspec()

    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()
    except Exception:
        pass

    try:
        import pymorphy2

        return pymorphy2.MorphAnalyzer()
    except Exception:
        pass

    find_and_attach_local_pymorphy2_paths(project_root)

    try:
        import pymorphy2

        return pymorphy2.MorphAnalyzer()
    except Exception:
        return None


def clean_html(raw_html: str) -> str:
    """удаляет базовую HTML-разметку и оставляет текст"""
    text = SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = COMMENT_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def extract_terms(text: str) -> set[str]:
    """извлекает уникальные термы из текста"""
    return {
        term
        for term in (match.group(0).lower() for match in TERM_RE.finditer(text))
        if len(term) >= 2
    }


def normalize_term(term: str, morph: Any | None, lemma_cache: dict[str, str]) -> str:
    """приводит терм к лемме с кешированием результата"""
    token = term.lower()
    cached = lemma_cache.get(token)
    if cached is not None:
        return cached

    lemma = token
    if morph is not None:
        try:
            lemma = morph.parse(token)[0].normal_form.lower()
        except Exception:
            lemma = token

    lemma_cache[token] = lemma
    return lemma


def build_inverted_index(
    pages_dir: Path,
    morph: Any | None,
    lemma_cache: dict[str, str],
) -> tuple[dict[str, list[str]], set[str]]:
    """строит лемматизированный инвертированный индекс: lemma -> sorted doc_ids"""
    if not pages_dir.exists():
        raise FileNotFoundError(f"Не найдена директория с документами: {pages_dir}")

    postings: dict[str, set[str]] = defaultdict(set)
    all_docs: set[str] = set()

    for page_file in sorted(pages_dir.glob("*.txt")):
        doc_id = page_file.stem
        all_docs.add(doc_id)

        raw_html = page_file.read_text(encoding="utf-8", errors="ignore")
        text = clean_html(raw_html)
        terms = extract_terms(text)

        lemmas_in_doc = {
            normalize_term(term, morph, lemma_cache)
            for term in terms
        }

        for lemma in lemmas_in_doc:
            postings[lemma].add(doc_id)

    inverted_index = {lemma: sorted(doc_ids) for lemma, doc_ids in postings.items()}
    return inverted_index, all_docs


def save_inverted_index(index: dict[str, list[str]], output_path: Path) -> None:
    """сохраняет индекс в формате: <lemma> <doc1> ... <docN>"""
    lines = []
    for lemma in sorted(index):
        lines.append(f"{lemma} {' '.join(index[lemma])}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def tokenize_query(query: str) -> list[str]:
    """разбирает строку запроса в токены и валидирует недопустимые фрагменты"""
    tokens: list[str] = []
    cursor = 0

    for match in QUERY_TOKEN_RE.finditer(query):
        gap = query[cursor:match.start()]
        if gap.strip():
            raise ValueError(f"Недопустимый фрагмент в запросе: {gap.strip()}")
        tokens.append(match.group(0))
        cursor = match.end()

    tail = query[cursor:]
    if tail.strip():
        raise ValueError(f"Недопустимый фрагмент в запросе: {tail.strip()}")

    if not tokens:
        raise ValueError("Пустой запрос")

    normalized: list[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in OPERATORS:
            normalized.append(upper)
        elif token in {"(", ")"}:
            normalized.append(token)
        else:
            normalized.append(token.lower())
    return normalized


def is_term(token: str) -> bool:
    """проверяет, является ли токен термом, а не оператором/скобкой"""
    return token not in OPERATORS and token not in {"(", ")"}


def lemmatize_query_terms(
    tokens: list[str],
    morph: Any | None,
    lemma_cache: dict[str, str],
) -> list[str]:
    """лемматизирует только термы запроса; операторы и скобки не меняет"""
    result: list[str] = []
    for token in tokens:
        if is_term(token):
            result.append(normalize_term(token, morph, lemma_cache))
        else:
            result.append(token)
    return result


def add_implicit_and(tokens: list[str]) -> list[str]:
    """добавляет неявный AND между соседними термами/скобками"""
    if not tokens:
        return tokens

    result: list[str] = [tokens[0]]
    for token in tokens[1:]:
        prev = result[-1]
        need_and = (
            (is_term(prev) or prev == ")")
            and (is_term(token) or token == "(" or token == "NOT")
        )
        if need_and:
            result.append("AND")
        result.append(token)
    return result


def to_postfix(tokens: list[str]) -> list[str]:
    """переводит инфиксный булев запрос в постфиксную форму"""
    output: list[str] = []
    stack: list[str] = []
    expect_operand = True

    for token in tokens:
        if is_term(token):
            if not expect_operand:
                raise ValueError(f"Ожидался оператор перед '{token}'")
            output.append(token)
            expect_operand = False
            continue

        if token == "(":
            if not expect_operand:
                raise ValueError("Ожидался оператор перед '('")
            stack.append(token)
            expect_operand = True
            continue

        if token == ")":
            if expect_operand:
                raise ValueError("Пустые скобки или оператор перед ')'")
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Несогласованные скобки: лишняя ')'")
            stack.pop()
            expect_operand = False
            continue

        if token == "NOT":
            if not expect_operand:
                raise ValueError("Оператор NOT стоит в неверной позиции")
            while stack and stack[-1] != "(" and PRECEDENCE[stack[-1]] > PRECEDENCE[token]:
                output.append(stack.pop())
            stack.append(token)
            expect_operand = True
            continue

        if expect_operand:
            raise ValueError(f"Ожидался терм/NOT/'(', найден оператор '{token}'")

        while stack and stack[-1] != "(" and (
            PRECEDENCE[stack[-1]] > PRECEDENCE[token]
            or (PRECEDENCE[stack[-1]] == PRECEDENCE[token] and token not in RIGHT_ASSOC)
        ):
            output.append(stack.pop())
        stack.append(token)
        expect_operand = True

    if expect_operand:
        raise ValueError("Запрос не может заканчиваться оператором")

    while stack:
        top = stack.pop()
        if top == "(":
            raise ValueError("Несогласованные скобки: отсутствует ')'")
        output.append(top)

    return output


def evaluate_postfix(
    postfix: list[str],
    index: dict[str, list[str]],
    all_docs: set[str],
) -> list[str]:
    """вычисляет постфиксный булев запрос над индексом"""
    stack: list[set[str]] = []

    for token in postfix:
        if is_term(token):
            stack.append(set(index.get(token, [])))
            continue

        if token == "NOT":
            if not stack:
                raise ValueError("Некорректный запрос: NOT без операнда")
            operand = stack.pop()
            stack.append(all_docs - operand)
            continue

        if len(stack) < 2:
            raise ValueError(f"Некорректный запрос: оператор '{token}' без двух операндов")

        right = stack.pop()
        left = stack.pop()

        if token == "AND":
            stack.append(left & right)
        elif token == "OR":
            stack.append(left | right)
        else:
            raise ValueError(f"Неизвестный оператор: {token}")

    if len(stack) != 1:
        raise ValueError("Некорректный запрос: ошибка структуры выражения")

    return sorted(stack[0])


def boolean_search(
    query: str,
    index: dict[str, list[str]],
    all_docs: set[str],
    morph: Any | None,
    lemma_cache: dict[str, str],
) -> list[str]:
    """полный цикл: токенизация -> лемматизация -> постфикс -> вычисление"""
    tokens = tokenize_query(query)
    tokens = lemmatize_query_terms(tokens, morph, lemma_cache)
    tokens = add_implicit_and(tokens)
    postfix = to_postfix(tokens)
    return evaluate_postfix(postfix, index, all_docs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построение инвертированного индекса по леммам и булев поиск (AND/OR/NOT)."
    )
    parser.add_argument(
        "--pages-dir",
        default="./crawl_output/pages",
        help="Путь к директории документов (*.txt)",
    )
    parser.add_argument(
        "--index-output",
        default="inverted_index.txt",
        help="Файл для сохранения инвертированного индекса",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Строка булевого запроса. Если не указано, запрос вводится вручную.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pages_dir = Path(args.pages_dir).resolve()
    index_output = Path(args.index_output).resolve()

    morph = get_morph_analyzer(project_root)
    lemma_cache: dict[str, str] = {}

    index, all_docs = build_inverted_index(pages_dir, morph, lemma_cache)
    save_inverted_index(index, index_output)

    print(f"Документов обработано: {len(all_docs)}")
    print(f"Уникальных лемм в индексе: {len(index)}")
    print(f"Индекс сохранён: {index_output}")
    if morph is None:
        print("Внимание: pymorphy не найден, использован режим без морфоанализа.")

    query = args.query if args.query is not None else input("Введите булев запрос: ").strip()
    if not query:
        print("Поиск пропущен: пустая строка запроса.")
        return

    result_docs = boolean_search(query, index, all_docs, morph, lemma_cache)
    print(f"Запрос: {query}")
    print(f"Найдено документов: {len(result_docs)}")
    if result_docs:
        print("Документы:", " ".join(result_docs))
    else:
        print("Совпадений нет.")


if __name__ == "__main__":
    main()
