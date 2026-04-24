const ROWS = 6;
const COLS = 7;
const EMPTY = 0;
const P1 = 1;
const P2 = 2;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const newHumanFirstBtn = document.getElementById("new-human-first");
const newAiFirstBtn = document.getElementById("new-ai-first");

let board = createEmptyBoard();
let done = false;
let winner = null;
let currentPlayer = P1;
let humanPiece = P1;
let aiPiece = P2;
let session = null;

function createEmptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));
}

function legalColumns() {
  const cols = [];
  for (let col = 0; col < COLS; col += 1) {
    if (board[0][col] === EMPTY) cols.push(col);
  }
  return cols;
}

function dropPiece(column, piece) {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][column] === EMPTY) {
      board[row][column] = piece;
      return row;
    }
  }
  return -1;
}

function hasConnect4(piece) {
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      if (board[row][col] !== piece) continue;
      if (
        col + 3 < COLS &&
        board[row][col + 1] === piece &&
        board[row][col + 2] === piece &&
        board[row][col + 3] === piece
      ) return true;
      if (
        row + 3 < ROWS &&
        board[row + 1][col] === piece &&
        board[row + 2][col] === piece &&
        board[row + 3][col] === piece
      ) return true;
      if (
        row + 3 < ROWS &&
        col + 3 < COLS &&
        board[row + 1][col + 1] === piece &&
        board[row + 2][col + 2] === piece &&
        board[row + 3][col + 3] === piece
      ) return true;
      if (
        row - 3 >= 0 &&
        col + 3 < COLS &&
        board[row - 1][col + 1] === piece &&
        board[row - 2][col + 2] === piece &&
        board[row - 3][col + 3] === piece
      ) return true;
    }
  }
  return false;
}

function buildObsForAi() {
  const obs = new Float32Array(2 * ROWS * COLS);
  let idx = 0;
  for (let ch = 0; ch < 2; ch += 1) {
    for (let row = 0; row < ROWS; row += 1) {
      for (let col = 0; col < COLS; col += 1) {
        const value = board[row][col];
        if (ch === 0) obs[idx] = value === aiPiece ? 1 : 0;
        else obs[idx] = value === humanPiece ? 1 : 0;
        idx += 1;
      }
    }
  }
  return obs;
}

async function chooseAiColumn() {
  const legal = legalColumns();
  if (legal.length === 0) return -1;

  const obs = new ort.Tensor("float32", buildObsForAi(), [1, 2, ROWS, COLS]);
  const maskData = new Uint8Array(COLS);
  for (const col of legal) maskData[col] = 1;
  const mask = new ort.Tensor("bool", maskData, [1, COLS]);
  const outputs = await session.run({ obs, action_masks: mask });
  const logits = outputs.logits.data;

  let bestCol = legal[0];
  let bestLogit = Number.NEGATIVE_INFINITY;
  for (const col of legal) {
    if (logits[col] > bestLogit) {
      bestLogit = logits[col];
      bestCol = col;
    }
  }
  return bestCol;
}

function updateStatus(text) {
  statusEl.textContent = text;
}

function gameStateText() {
  if (done) {
    if (winner === humanPiece) return "You win.";
    if (winner === aiPiece) return "AI wins.";
    return "Draw.";
  }
  if (!session) return "Loading ONNX model...";
  return currentPlayer === humanPiece ? "Your turn." : "AI is thinking...";
}

function render() {
  boardEl.innerHTML = "";
  const legal = new Set(legalColumns());
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const value = board[row][col];
      const btn = document.createElement("button");
      btn.className = "cell";
      if (value === humanPiece) btn.classList.add("human");
      if (value === aiPiece) btn.classList.add("ai");
      const humanTurn = !done && currentPlayer === humanPiece;
      if (humanTurn && legal.has(col)) {
        btn.classList.add("playable");
        btn.disabled = false;
        btn.title = `Drop in column ${col + 1}`;
        btn.addEventListener("click", () => humanMove(col));
      } else {
        btn.disabled = true;
      }
      boardEl.appendChild(btn);
    }
  }
  updateStatus(gameStateText());
}

function applyMove(column, piece) {
  const row = dropPiece(column, piece);
  if (row < 0) return false;
  if (hasConnect4(piece)) {
    done = true;
    winner = piece;
  } else if (legalColumns().length === 0) {
    done = true;
    winner = null;
  } else {
    currentPlayer = piece === P1 ? P2 : P1;
  }
  return true;
}

async function maybeAiMove() {
  if (done || currentPlayer !== aiPiece || !session) return;
  render();
  const col = await chooseAiColumn();
  if (col >= 0) applyMove(col, aiPiece);
  render();
}

async function humanMove(column) {
  if (done || currentPlayer !== humanPiece) return;
  if (!legalColumns().includes(column)) return;
  applyMove(column, humanPiece);
  render();
  await maybeAiMove();
}

async function newGame(humanStarts) {
  board = createEmptyBoard();
  done = false;
  winner = null;
  humanPiece = humanStarts ? P1 : P2;
  aiPiece = humanStarts ? P2 : P1;
  currentPlayer = P1;
  render();
  await maybeAiMove();
}

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./policy.onnx");
    await newGame(true);
  } catch (error) {
    updateStatus(`Failed to load policy.onnx: ${error.message}`);
    render();
  }
}

newHumanFirstBtn.addEventListener("click", () => newGame(true));
newAiFirstBtn.addEventListener("click", () => newGame(false));
render();
loadModel();
