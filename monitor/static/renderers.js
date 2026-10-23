// Episode renderers, one per environment, drawing on a canvas in the dashboard's
// palette.
//
// A training run logs Episode(env, **arrays) (monitor/client.py); the dashboard
// fetches the arrays (episodes.js loadArrays) and calls RENDERERS[env](arrays) for a
// view of them:
//
//   {
//     width, height,       // its size, in the units draw() draws in
//     frames, fps,         // how many frames, and how many a second at 1x
//     draw(ctx, frame),    // draw a frame
//     counter(frame),      // optional: the frame, as the player's counter says it
//     speeds, loop,        // optional: the speeds offered, and false to stop at the end
//     players,             // optional: for a two-player game, [{name, color, ring}]
//                          // by colour, the first to move in its usual opening first
//     pieces(frame, colour),  // optional: how many pieces of a colour are on the board
//   }
//
// Arrays are NdArrays (episodes.js), floats as float32. A view should do any
// per-episode work when it's made, not every frame.
//
// The board games' arrays are monitor/episodes.py's pgx_two_player_episode:
// board (frames, rows, cols) of 0 for empty, 1 and 2 for the two colours; to_move
// (frames,), the colour to move; agent, the trained agent's colour; values (frames, 2),
// the agent's and its opponent's value estimates; completed and outcome, the agent's
// result once the game is over.

import { BLUE, EDGE, GOLD, INK, INK_STRONG, MID, ORANGE, RED, SUNKEN, WELL } from "./palette.js";

export const RENDERERS = {
  othello,
  connect_four: connectFour,
  tic_tac_toe: ticTacToe,
};

function disc(ctx, x, y, r, fill, stroke = null, width = 1) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}

function line(ctx, [x0, y0], [x1, y1], color, width) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

// ---------------------------------------------------------------- board games

// a move every 0.9s at 1x
const MOVE_FPS = 1 / 0.9;
const MOVE_SPEEDS = [0.5, 1, 2, 4];

// What's common to the board games: the board at each frame, the cell each move
// filled, lines of `connect` in a row, and the view's counter and speeds.
function boardGame({ board }, { connect = null, passes = false } = {}) {
  const [frames, rows, cols] = board.shape;
  const cell = (t, r, c) => board.get(t, r, c);
  // the cell the move into frame t filled, [r, c], or null (the start, a pass);
  // othello's moves turn other discs over, but only fill the one
  const moves = [null];
  for (let t = 1; t < frames; t++) {
    let move = null;
    for (let r = 0; r < rows && !move; r++) {
      for (let c = 0; c < cols; c++) {
        if (!cell(t - 1, r, c) && cell(t, r, c)) {
          move = [r, c];
          break;
        }
      }
    }
    moves.push(move);
  }
  // the cells of `connect` in a row through the last move, if it made one
  const win = connect && winningLine(cell, frames - 1, moves.at(-1), connect, rows, cols);
  return {
    frames,
    rows,
    cols,
    cell,
    moves,
    // the line shows from the move that made it on
    win: (t) => (t === frames - 1 ? win : null),
    fps: MOVE_FPS,
    speeds: MOVE_SPEEDS,
    loop: false,
    counter(t) {
      return `move ${t}/${frames - 1}${passes && !moves[t] ? " pass" : ""}`;
    },
  };
}

function winningLine(cell, t, move, n, rows, cols) {
  if (!move) return null;
  const [r0, c0] = move;
  const color = cell(t, r0, c0);
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    // from the far end of the run through the move
    let r = r0, c = c0;
    while (r - dr >= 0 && r - dr < rows && c - dc >= 0 && c - dc < cols && cell(t, r - dr, c - dc) === color) {
      r -= dr;
      c -= dc;
    }
    const run = [];
    while (r >= 0 && r < rows && c >= 0 && c < cols && cell(t, r, c) === color) {
      run.push([r, c]);
      r += dr;
      c += dc;
    }
    if (run.length >= n) return run;
  }
  return null;
}

// coordinates along the edges, as players write moves: files a-h, ranks 1-8
function labels(ctx, game, at, size, pad) {
  ctx.fillStyle = MID;
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let c = 0; c < game.cols; c++) ctx.fillText(String.fromCharCode(97 + c), at(0, c)[0], pad / 2);
  for (let r = 0; r < game.rows; r++) ctx.fillText(String(r + 1), pad / 2, at(r, 0)[1]);
}

// ---------------------------------------------------------------- othello

const OTHELLO_CELL = 40, OTHELLO_PAD = 18;
const OTHELLO_BOARD = "#2f6f52";

function othello(arrays) {
  const game = boardGame(arrays, { passes: true });
  const size = OTHELLO_CELL * 8;
  const W = OTHELLO_PAD + size + 6, H = OTHELLO_PAD + size + 6;
  const at = (r, c) => [OTHELLO_PAD + (c + 0.5) * OTHELLO_CELL, OTHELLO_PAD + (r + 0.5) * OTHELLO_CELL];
  const fills = [INK_STRONG, WELL];
  return {
    ...game,
    width: W,
    height: H,
    players: [
      { name: "Black", color: INK_STRONG },
      { name: "White", color: WELL },
    ],
    // the score
    pieces(t, color) {
      let n = 0;
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) n += game.cell(t, r, c) === color + 1;
      return n;
    },
    draw(ctx, t) {
      ctx.fillStyle = SUNKEN;
      ctx.fillRect(0, 0, W, H);
      labels(ctx, game, at, size, OTHELLO_PAD);
      ctx.fillStyle = OTHELLO_BOARD;
      ctx.fillRect(OTHELLO_PAD, OTHELLO_PAD, size, size);
      for (let i = 0; i <= 8; i++) {
        const x = OTHELLO_PAD + i * OTHELLO_CELL;
        line(ctx, [x, OTHELLO_PAD], [x, OTHELLO_PAD + size], INK_STRONG, 1);
        line(ctx, [OTHELLO_PAD, x], [OTHELLO_PAD + size, x], INK_STRONG, 1);
      }
      // the four dots that mark out the centre
      for (const [r, c] of [[2, 2], [2, 6], [6, 2], [6, 6]]) {
        disc(ctx, OTHELLO_PAD + c * OTHELLO_CELL, OTHELLO_PAD + r * OTHELLO_CELL, 3, INK_STRONG);
      }
      const r = OTHELLO_CELL * 0.4;
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const v = game.cell(t, row, col);
          if (v) disc(ctx, ...at(row, col), r, fills[v - 1], INK_STRONG, 1);
        }
      }
      const move = game.moves[t];
      if (move) disc(ctx, ...at(...move), 4, RED);
    },
  };
}

// ---------------------------------------------------------------- connect four

const C4_CELL = 46, C4_PAD = 8;

function connectFour(arrays) {
  const game = boardGame(arrays, { connect: 4 });
  const W = C4_PAD * 2 + 7 * C4_CELL, H = C4_PAD * 2 + 6 * C4_CELL;
  const at = (r, c) => [C4_PAD + (c + 0.5) * C4_CELL, C4_PAD + (r + 0.5) * C4_CELL];
  const fills = [RED, GOLD];
  const R = C4_CELL * 0.38;
  return {
    ...game,
    width: W,
    height: H,
    players: [
      { name: "Red", color: RED },
      { name: "Yellow", color: GOLD },
    ],
    draw(ctx, t) {
      ctx.fillStyle = BLUE;
      ctx.beginPath();
      ctx.roundRect(0, 0, W, H, 10);
      ctx.fill();
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 7; c++) {
          const v = game.cell(t, r, c);
          disc(ctx, ...at(r, c), R, v ? fills[v - 1] : WELL);
        }
      }
      const move = game.moves[t];
      if (move) disc(ctx, ...at(...move), 5, WELL);
      const win = game.win(t);
      if (win) for (const rc of win) disc(ctx, ...at(...rc), R, null, INK, 4);
    },
  };
}

// ---------------------------------------------------------------- tic-tac-toe

const TTT_CELL = 80, TTT_PAD = 12;

function ticTacToe(arrays) {
  const game = boardGame(arrays, { connect: 3 });
  const size = 3 * TTT_CELL;
  const W = TTT_PAD * 2 + size, H = TTT_PAD * 2 + size;
  const at = (r, c) => [TTT_PAD + (c + 0.5) * TTT_CELL, TTT_PAD + (r + 0.5) * TTT_CELL];
  const mark = TTT_CELL * 0.28;
  return {
    ...game,
    width: W,
    height: H,
    players: [
      { name: "X", color: BLUE },
      { name: "O", color: ORANGE, ring: true },
    ],
    draw(ctx, t) {
      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, W, H);
      const move = game.moves[t];
      if (move) {
        ctx.fillStyle = SUNKEN;
        ctx.fillRect(TTT_PAD + move[1] * TTT_CELL, TTT_PAD + move[0] * TTT_CELL, TTT_CELL, TTT_CELL);
      }
      ctx.lineCap = "round";
      for (let i = 1; i < 3; i++) {
        const x = TTT_PAD + i * TTT_CELL;
        line(ctx, [x, TTT_PAD + 4], [x, TTT_PAD + size - 4], EDGE, 3);
        line(ctx, [TTT_PAD + 4, x], [TTT_PAD + size - 4, x], EDGE, 3);
      }
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const v = game.cell(t, r, c);
          const [x, y] = at(r, c);
          if (v === 1) {
            line(ctx, [x - mark, y - mark], [x + mark, y + mark], BLUE, 7);
            line(ctx, [x - mark, y + mark], [x + mark, y - mark], BLUE, 7);
          } else if (v === 2) {
            ctx.beginPath();
            ctx.arc(x, y, mark, 0, 2 * Math.PI);
            ctx.strokeStyle = ORANGE;
            ctx.lineWidth = 7;
            ctx.stroke();
          }
        }
      }
      const win = game.win(t);
      if (win) {
        const [a, b] = [at(...win[0]), at(...win.at(-1))];
        // a little past the end cells' marks
        const dx = b[0] - a[0], dy = b[1] - a[1], k = (mark * 1.3) / Math.hypot(dx, dy);
        ctx.globalAlpha = 0.85;
        line(ctx, [a[0] - dx * k, a[1] - dy * k], [b[0] + dx * k, b[1] + dy * k], INK, 5);
        ctx.globalAlpha = 1;
      }
    },
  };
}
