// The search explorer: what each player's search explored from each position of a
// test game, browsed node by node.
//
// The arrays are monitor/episodes.py's _searches: one search per (position, player)
// with a tree, search_frame (the frame of the position it's of), search_player (0 for
// the trained agent, 1 for its opponent), search_offset (where its nodes start in the
// node arrays, and one past the end) and search_policy (its root's policy); then per
// node, node_parent (in its own search, -1 for the root), node_action, node_visits,
// node_value (for the player who chose it; the root's for its player to move),
// node_prior, node_terminal, node_board, node_to_move and node_legal.
//
// The explorer shows a position's board with a player's explored moves on it, and
// beside it each player's explored moves there, best first, each with the best line its
// search explored after it: hovering a move shows its position, clicking one goes there.

import { h } from "./dom.js";
import { BLUE, EDGE, INK_STRONG, ORANGE, WELL } from "./palette.js";

export const hasSearches = (arrays) => "search_offset" in arrays;

// the searches, indexed: each node's children, best first (by their value for the
// player choosing, node_value; the most visited of equal ones)
function index(arrays) {
  const offset = arrays.search_offset.data;
  const num = arrays.search_frame.length;
  const parent = arrays.node_parent.data;
  const visits = arrays.node_visits.data;
  const searchOf = new Int32Array(offset[num]);
  const children = Array.from({ length: offset[num] }, () => []);
  const byFrame = new Map();
  for (let s = 0; s < num; s++) {
    const frame = arrays.search_frame.get(s);
    if (!byFrame.has(frame)) byFrame.set(frame, []);
    byFrame.get(frame).push(s);
    for (let g = offset[s]; g < offset[s + 1]; g++) {
      searchOf[g] = s;
      if (parent[g] >= 0) children[offset[s] + parent[g]].push(g);
    }
  }
  const value = arrays.node_value.data;
  for (const list of children) list.sort((a, b) => value[b] - value[a] || visits[b] - visits[a]);
  const toMove = (g) => arrays.node_to_move.get(g);
  const up = (g) => (parent[g] < 0 ? -1 : offset[searchOf[g]] + parent[g]);
  const searched = [0, 1].map((p) => arrays.search_player.data.some((x) => x === p));
  return {
    byFrame,
    children,
    // whether player p searched anywhere in the episode
    searched: (p) => searched[p],
    root: (s) => offset[s],
    size: (s) => offset[s + 1] - offset[s],
    player: (s) => arrays.search_player.get(s),
    policy: (s, a) => arrays.search_policy.get(s, a),
    searchOf: (g) => searchOf[g],
    up,
    visits: (g) => visits[g],
    action: (g) => arrays.node_action.get(g),
    prior: (g) => arrays.node_prior.get(g),
    terminal: (g) => !!arrays.node_terminal.get(g),
    legal: (g, a) => !!arrays.node_legal.get(g, a),
    numActions: arrays.node_legal.shape[1],
    toMove,
    board: (g) => (r, c) => arrays.node_board.get(g, r, c),
    // a node's value for a colour: node_value is for the player who chose it
    valueFor(g, color) {
      const p = up(g);
      const chooser = p < 0 ? toMove(g) : toMove(p);
      const v = arrays.node_value.get(g);
      return chooser === color ? v : -v;
    },
    // the nodes from the root down to g
    path(g) {
      const path = [];
      for (; g >= 0; g = up(g)) path.unshift(g);
      return path;
    },
    // g, then the best child of each node after it, as far as the search went
    bestLine(g) {
      const line = [g];
      while (children[line.at(-1)].length) line.push(children[line.at(-1)][0]);
      return line;
    },
  };
}


// a side of a two-player game (renderers.js players), as a mark: its symbol, or a disc
export function sideMark(side) {
  const el = side.symbol
    ? h("b", { class: "mark-sym", text: side.symbol })
    : h("i", { class: `swatch${side.ring ? " ring" : ""}` });
  el.style.setProperty("--c", side.color);
  return el;
}

// a value for the first player (renderers.js players[0]): + favours it and − the
// other, whoever's value it is, then the mark of the side it favours (an even one, none)
export function advantage(v, players) {
  const size = Math.abs(v).toFixed(3);
  return Number(size)
    ? h("span", { class: "adv" }, `${v > 0 ? "+" : "−"}${size}`, sideMark(players[v > 0 ? 0 : 1]))
    : h("span", { class: "adv" }, size);
}

// how far apart two players' values of a position can be before it's flagged
const DISAGREE = 0.2;
const pct = (x) => `${Math.round(100 * x)}%`;

// a value from -1 to 1 as a colour, orange (bad for the player choosing) to blue (good)
function valueColor(v) {
  const [a, b] = v >= 0 ? [WELL, BLUE] : [WELL, ORANGE];
  // most values are near 0; the square root tells them apart
  const t = Math.sqrt(Math.min(1, Math.abs(v)));
  const mix = (i) => Math.round(parseInt(a.slice(i, i + 2), 16) * (1 - t) + parseInt(b.slice(i, i + 2), 16) * t);
  return `rgb(${mix(1)}, ${mix(3)}, ${mix(5)})`;
}

// keys go to the explorer under the pointer, unless they're being typed somewhere
let hovered = null;
document.addEventListener("keydown", (ev) => {
  if (!hovered || ev.target.closest?.("input, select, textarea")) return;
  if (hovered.key(ev.key)) ev.preventDefault();
});

// The explorer for an episode's searches, drawn with its view (renderers.js) on the
// episode's own board. who(p) describes player p (0 the trained agent, 1 its opponent)
// as {role, color, ring, colorIndex}: what it is, its colour as drawn and as an index
// into view.players.
//
// It explores positions, each the moves from a frame's position, and shows each
// player's tree at the position: the moves it explored there, in a section of its own,
// over a plot of each player's value of the game's positions so far (explorer.side),
// and its value of the position (explorer.evals). It gives the episode
// player a view to play (explorer.view), which draws each frame as the explorer has
// it: a new frame starts at its own position. explorer.attach(player) then lets the
// explorer redraw the player's board, take clicks on it, and pause the episode when
// it's used.
export function searchExplorer(view, arrays, { who }) {
  const tree = index(arrays);
  // the frame, the moves from its position to the one on the board, a position being
  // previewed (as its moves, or null), and the player whose search the board shows
  const st = { frame: -1, path: [], preview: null, shown: 0 };
  let player = null;

  const heading = h("div", { class: "explore-heading" });
  const evals = h("div", { class: "explore-evals" });
  const crumbs = h("div", { class: "explore-crumbs" });
  const summary = h("div", { class: "explore-summary" });
  const sections = h("div", { class: "explore-sections" });
  const plot = h("div", { class: "explore-plot" });
  const side = h("div", { class: "explore-side" }, heading, crumbs, summary, sections, plot);

  const interact = () => player?.pause();
  const redraw = () => player?.redraw();
  // values are all for the first player, whoever's tree they're from, so they compare
  // directly
  const value = (g) => tree.valueFor(g, 0);
  const shown = (g) => advantage(value(g), view.players);
  // the game's result once it's over, as a value for the first player (outcome is the
  // agent's): the value of its final position, which nobody searches; else null
  const terminal = arrays.completed?.item() ? (who(0).colorIndex === 0 ? 1 : -1) * arrays.outcome.item() : null;
  const isEnd = (t) => terminal != null && t === view.frames - 1;
  // two values of a position too far apart
  const disagree = (a, b) => a >= 0 && b >= 0 && Math.abs(value(a) - value(b)) > DISAGREE;
  // each player's search of the frame's position, [agent's, opponent's], null for none
  const searchesAt = (t) => [0, 1].map((p) => (tree.byFrame.get(t) ?? []).find((s) => tree.player(s) === p) ?? null);
  const childFor = (n, a) => (n < 0 ? -1 : (tree.children[n].find((c) => tree.action(c) === a) ?? -1));
  // the node a position is in a search, -1 where the search doesn't reach it
  function nodeAt(s, path) {
    if (s == null) return -1;
    let n = tree.root(s);
    for (const a of path) n = childFor(n, a);
    return n;
  }
  // a position's node in each player's search of the frame's, [agent's, opponent's]
  const nodesAt = (path) => searchesAt(st.frame).map((s) => nodeAt(s, path));
  // a position's node in either search, for its board
  const anyNode = (path) => nodesAt(path).find((n) => n >= 0) ?? -1;
  const pathKey = (path) => path.join(",");
  // the cell a position's last move filled; the root's is the move into its frame
  function moveCell(path) {
    if (!path.length) return view.moves[st.frame];
    const before = anyNode(path.slice(0, -1));
    return before < 0 ? null : view.actionCell(path.at(-1), tree.board(before));
  }

  // ------------------------------------------------------------ the board

  function draw(ctx) {
    const path = st.preview ?? st.path;
    const g = anyNode(path);
    if (g < 0) {
      view.drawBoard(ctx, view.board(st.frame), view.moves[st.frame]);
      return;
    }
    view.drawBoard(ctx, tree.board(g), moveCell(path));
    if (st.preview) return;
    // the shown player's explored moves, sized by their share of its visits, coloured
    // by their value for the player choosing; the moves it never explored, faintly
    const n = nodesAt(path)[st.shown] >= 0 ? nodesAt(path)[st.shown] : g;
    const kids = tree.children[n];
    const total = kids.reduce((k, c) => k + tree.visits(c), 0) || 1;
    const explored = new Set();
    for (const c of kids) {
      explored.add(tree.action(c));
      const cell = view.actionCell(tree.action(c), tree.board(n));
      if (!cell) continue;
      const [x, y] = view.at(...cell);
      const share = tree.visits(c) / total;
      const r = view.cellSize * (0.16 + 0.26 * Math.sqrt(share));
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = valueColor(tree.valueFor(c, tree.toMove(n)));
      ctx.globalAlpha = 0.92;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = c === kids[0] ? INK_STRONG : EDGE;
      ctx.lineWidth = c === kids[0] ? 2 : 1;
      ctx.stroke();
      ctx.fillStyle = INK_STRONG;
      ctx.font = `600 ${Math.round(view.cellSize * 0.27)}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(tree.visits(c)), x, y + 0.5);
    }
    for (let a = 0; a < tree.numActions; a++) {
      if (explored.has(a) || !tree.legal(n, a)) continue;
      const cell = view.actionCell(a, tree.board(n));
      if (!cell) continue;
      const [x, y] = view.at(...cell);
      ctx.beginPath();
      ctx.arc(x, y, view.cellSize * 0.12, 0, 2 * Math.PI);
      ctx.strokeStyle = EDGE;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // the position after the move under the pointer on the board, if either player
  // explored it, or null
  function pathAt(ev) {
    const nodes = nodesAt(st.path);
    const rect = ev.target.getBoundingClientRect();
    const a = view.actionAt(((ev.clientX - rect.left) / rect.width) * view.width, ((ev.clientY - rect.top) / rect.height) * view.height);
    return a != null && nodes.some((n) => childFor(n, a) >= 0) ? [...st.path, a] : null;
  }

  // ------------------------------------------------------------ navigation

  function preview(path) {
    if ((path && pathKey(path)) === (st.preview && pathKey(st.preview))) return;
    st.preview = path;
    const key = path && pathKey(path);
    for (const t of sections.querySelectorAll("[data-path]")) t.classList.toggle("hot", t.dataset.path === key);
    renderEvals();
    redraw();
  }

  function go(path) {
    interact();
    st.path = path;
    st.preview = null;
    renderPanels();
    redraw();
  }

  function show(p) {
    interact();
    st.shown = p;
    renderPanels();
    redraw();
  }

  // frame t's position, its explored moves on the board the ones of the player who
  // chose its move
  function setFrame(t) {
    st.frame = t;
    st.path = [];
    st.preview = null;
    const searches = searchesAt(t);
    const toMove = arrays.to_move?.get(t);
    const searched = [0, 1].filter((p) => searches[p] != null);
    st.shown = searched.find((p) => who(p).colorIndex === toMove) ?? searched[0] ?? 0;
    renderPanels();
  }

  // ------------------------------------------------------------ the panels

  function renderPanels() {
    renderEvals();
    renderSide();
    renderPlot();
  }

  // each player's value of the game's positions, from its tree of each, up to the
  // frame shown: a line per player, + (the first player ahead) up; clicked, the game
  // goes to that move
  const PLOT_W = 520, PLOT_H = 130, PAD_L = 26, PAD_R = 8, PAD_Y = 10;
  const rootValues = [0, 1].map((p) =>
    Array.from({ length: view.frames }, (_, t) => {
      const s = (tree.byFrame.get(t) ?? []).find((x) => tree.player(x) === p);
      // the final position's value is the result
      if (s == null) return isEnd(t) ? terminal : null;
      return value(tree.root(s));
    }),
  );
  const plotX = (t) => PAD_L + (t / Math.max(1, view.frames - 1)) * (PLOT_W - PAD_L - PAD_R);
  const plotY = (v) => PLOT_H / 2 - v * (PLOT_H / 2 - PAD_Y);
  const SERIES = ["var(--s1)", "var(--s2)"];
  function renderPlot() {
    const t = st.frame;
    const svg = h("svg", { viewBox: `0 0 ${PLOT_W} ${PLOT_H}`, class: "plot" });
    for (const v of [1, 0, -1]) {
      svg.append(h("line", { class: v ? "grid" : "axis", x1: PAD_L, x2: PLOT_W - PAD_R, y1: plotY(v), y2: plotY(v) }));
    }
    svg.append(
      h("text", { class: "tick", x: PAD_L - 6, y: plotY(1) + 3, "text-anchor": "end", text: "+1" }),
      h("text", { class: "tick", x: PAD_L - 6, y: plotY(0) + 3, "text-anchor": "end", text: "0" }),
      h("text", { class: "tick", x: PAD_L - 6, y: plotY(-1) + 3, "text-anchor": "end", text: "−1" }),
      h("line", { class: "cursor", x1: plotX(t), x2: plotX(t), y1: PAD_Y / 2, y2: PLOT_H - PAD_Y / 2 }),
    );
    for (const p of [0, 1]) {
      if (!tree.searched(p)) continue;
      let d = "", on = false;
      for (let f = 0; f <= t; f++) {
        const v = rootValues[p][f];
        if (v == null) {
          on = false;
          continue;
        }
        d += `${on ? "L" : "M"}${plotX(f).toFixed(1)},${plotY(v).toFixed(1)}`;
        on = true;
      }
      svg.append(h("path", { class: "line", d, stroke: SERIES[p] }));
      const v = rootValues[p][t];
      if (v != null) svg.append(h("circle", { cx: plotX(t), cy: plotY(v), r: 3, fill: SERIES[p] }));
    }
    svg.addEventListener("click", (ev) => {
      const rect = svg.getBoundingClientRect();
      const x = ((ev.clientX - rect.left) / rect.width) * PLOT_W;
      const f = Math.round(((x - PAD_L) / (PLOT_W - PAD_L - PAD_R)) * (view.frames - 1));
      interact();
      player?.setFrame(Math.min(Math.max(f, 0), view.frames - 1));
    });
    const legend = [0, 1]
      .filter((p) => tree.searched(p))
      .map((p) => {
        const key = h("i", { class: "key" });
        key.style.background = SERIES[p];
        return h("span", { class: "legend-key" }, key, sideMark(who(p)), who(p).role);
      });
    plot.replaceChildren(h("div", { class: "plot-legend" }, ...legend), svg);
  }

  // a row per player: its value of the position on the board, from its tree, or at
  // the end the game's result; clicked, its explored moves are the ones on the board
  function renderEvals() {
    const searches = searchesAt(st.frame);
    const nodes = nodesAt(st.preview ?? st.path);
    const rows = [0, 1].map((p) => {
      const side = who(p);
      const cells = [sideMark(side), h("span", { class: "who", text: `${side.role} · ${side.name}` })];
      if (isEnd(st.frame)) {
        // the final position's value, the result, and each one's: the agent's outcome,
        // and its opponent's the other way round
        const outcome = arrays.outcome.item();
        const result = Math.sign(p === 0 ? outcome : -outcome);
        const text = result > 0 ? "won" : result < 0 ? "lost" : "drew";
        return h(
          "div",
          { class: "explore-eval over" },
          ...cells,
          h("b", { class: "num" }, advantage(terminal, view.players)),
          h("span", { class: `result ${text}`, text }),
        );
      }
      const n = nodes[p];
      const val = n < 0 ? h("span", { class: "unknown", text: tree.searched(p) ? "???" : "–" }) : shown(n);
      const bang = h("span", { class: "disagree", text: p !== st.shown && disagree(nodes[0], nodes[1]) ? "!" : "" });
      if (searches[p] == null) return h("div", { class: "explore-eval" }, ...cells, h("b", { class: "num" }, val), bang);
      return h("button", { class: `explore-eval${p === st.shown ? " on" : ""}`, onclick: () => show(p) }, ...cells, h("b", { class: "num" }, val), bang);
    });
    evals.replaceChildren(...rows);
  }

  function renderSide() {
    // the move to be chosen at the position: the game's next, plus the moves into it
    heading.textContent = `move ${st.frame + st.path.length + 1}`;
    if (isEnd(st.frame)) {
      // the final position: nothing searched, so each player's section is empty
      heading.textContent = "game over";
      crumbs.replaceChildren();
      summary.replaceChildren();
      sections.replaceChildren(
        ...[0, 1]
          .filter((p) => tree.searched(p))
          .map((p) => h("div", { class: "explore-sec" }, sectionHeader(p, "", advantage(terminal, view.players)), h("div", { class: "explore-lines" }, columns()))),
      );
      return;
    }
    const g = anyNode(st.path);
    if (g < 0) {
      crumbs.replaceChildren();
      summary.replaceChildren(h("div", { class: "dim", text: tree.byFrame.has(st.frame) ? "" : "game over" }));
      sections.replaceChildren();
      return;
    }
    // the moves from the root, each a way back up
    crumbs.replaceChildren(
      h("button", { class: `crumb${st.path.length ? "" : " on"}`, onclick: () => go([]), text: "root" }),
      ...st.path.flatMap((a, i) => [
        h("span", { class: "sep", text: "›" }),
        h("button", { class: `crumb${i === st.path.length - 1 ? " on" : ""}`, onclick: () => go(st.path.slice(0, i + 1)), text: view.moveName(a) }),
      ]),
    );
    const mover = view.players?.[tree.toMove(g)];
    summary.replaceChildren(h("span", { text: tree.terminal(g) ? "game over" : `${mover?.name ?? "player"} to move` }));
    const nodes = nodesAt(st.path);
    sections.replaceChildren(...[0, 1].filter((p) => tree.searched(p)).map((p) => section(p, nodes)));
  }

  // a player's section's header: which player, then its visits and value of the
  // position; clicked, its explored moves are the ones on the board
  function sectionHeader(p, visits, val) {
    const side = who(p);
    return h(
      "button",
      { class: `explore-sec-hdr${p === st.shown ? " on" : ""}`, onclick: () => show(p) },
      sideMark(side),
      h("span", { class: "who", text: `${side.role} · ${side.name}` }),
      h("span", { class: "num dim", text: visits }),
      h("b", { class: "num" }, val),
    );
  }

  // the labels of a section's columns
  const columns = () =>
    h(
      "div",
      { class: "explore-cols" },
      h("span"),
      h("span", { class: "col", text: "visits" }),
      h("span", { class: "col", text: "value" }),
      h("span"),
      h("span", { class: "col", text: "prior" }),
    );

  // a player's tree at the position: its visits and value of it, then a row per move
  // it explored, best first: its visits (and share), value, prior and the best line
  // after it, flagged where the other player's value of the move disagrees; then the
  // legal moves it never explored, greyed out, with their priors where it has them
  // (its root's policy), most likely first
  function section(p, nodes) {
    const n = nodes[p];
    const other = nodes[1 - p];
    const header = sectionHeader(p, n < 0 ? "" : `${tree.visits(n)} visits`, n < 0 ? h("span", { class: "unknown", text: "???" }) : shown(n));
    if (n < 0) return h("div", { class: "explore-sec" }, header);
    const cols = columns();
    const kids = tree.children[n];
    const total = kids.reduce((k, c) => k + tree.visits(c), 0);
    const rows = kids.map((c) => {
      const line = tree.bestLine(c);
      const share = total ? tree.visits(c) / total : 0;
      const bar = h("i", { class: "bar" });
      bar.style.width = pct(share);
      const to = (i) => [...st.path, ...line.slice(0, i + 1).map((m) => tree.action(m))];
      const tokens = line.map((m, i) =>
        h("button", {
          class: `tok${i === 0 ? " first" : ""}`,
          "data-path": pathKey(to(i)),
          text: view.moveName(tree.action(m)),
          onpointerenter: () => preview(to(i)),
          onclick: (ev) => (ev.stopPropagation(), go(to(i))),
        }),
      );
      if (tree.terminal(line.at(-1))) tokens.push(h("span", { class: "end", text: "end" }));
      return h(
        "div",
        { class: "explore-line", onpointerleave: () => preview(null) },
        h(
          "div",
          { class: "stats", onpointerenter: () => preview(to(0)), onclick: () => go(to(0)) },
          bar,
          h("b", { class: "mv", text: view.moveName(tree.action(c)) }),
          h("span", { class: "num" }, `${tree.visits(c)} `, h("span", { class: "dim", text: `(${pct(share)})` })),
          h("span", { class: "num val" }, shown(c)),
          h("span", { class: "disagree", text: disagree(c, childFor(other, tree.action(c))) ? "!" : "" }),
          h("span", { class: "prior", text: tree.prior(c).toFixed(2) }),
        ),
        h("div", { class: "pv" }, tokens),
      );
    });
    const explored = new Set(kids.map((c) => tree.action(c)));
    const rest = [];
    for (let a = 0; a < tree.numActions && !tree.terminal(n); a++) {
      if (!explored.has(a) && tree.legal(n, a)) rest.push({ a, prior: st.path.length ? null : tree.policy(searchesAt(st.frame)[p], a) });
    }
    rest.sort((x, y) => (y.prior ?? 0) - (x.prior ?? 0));
    for (const { a, prior } of rest) {
      rows.push(
        h(
          "div",
          { class: "explore-line unexplored" },
          h(
            "div",
            { class: "stats" },
            h("b", { class: "mv", text: view.moveName(a) }),
            h("span", { class: "num", text: "0" }),
            h("span"),
            h("span"),
            h("span", { class: "prior", text: prior == null ? "" : prior.toFixed(2) }),
          ),
        ),
      );
    }
    return h("div", { class: "explore-sec" }, header, h("div", { class: "explore-lines" }, cols, ...rows));
  }

  // ------------------------------------------------------------ the explorer

  const explorer = {
    side,
    evals,
    // the episode's frames, each drawn as the explorer has it: a new frame starts at
    // its own position
    view: {
      ...view,
      draw(ctx, t) {
        if (t !== st.frame) setFrame(t);
        draw(ctx);
      },
    },
    attach(p) {
      player = p;
      // explored moves on the board are previewed under the pointer, and clicked, gone to
      p.canvas.addEventListener("pointermove", (ev) => preview(pathAt(ev)));
      p.canvas.addEventListener("pointerleave", () => preview(null));
      p.onClick = (ev) => {
        const path = pathAt(ev);
        if (!path) return false;
        go(path);
        return true;
      };
    },
    key(key) {
      if (anyNode(st.path) < 0 || !side.isConnected) return false;
      if (key === "ArrowLeft" || key === "Backspace") {
        if (st.path.length) go(st.path.slice(0, -1));
      } else if (key === "ArrowRight") {
        // the shown player's best move, or the other's where it has none
        const nodes = nodesAt(st.path);
        const n = [nodes[st.shown], nodes[1 - st.shown]].find((m) => m >= 0 && tree.children[m].length);
        if (n != null) go([...st.path, tree.action(tree.children[n][0])]);
      } else if (key === "Escape") go([]);
      else return false;
      return true;
    },
  };
  side.addEventListener("pointerenter", () => (hovered = explorer));
  side.addEventListener("pointerleave", () => hovered === explorer && (hovered = null));
  return explorer;
}
