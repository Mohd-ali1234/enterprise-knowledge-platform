// Deterministic force-directed layout for the knowledge graph.
//
// React Flow positions nodes but does not lay them out, and pulling in a
// layout engine (dagre, elk) for one screen is not worth the weight. This is a
// compact spring-and-repulsion simulation:
//
//   * connected nodes attract along their edge (spring)
//   * every pair repels (Coulomb), which spreads clusters apart
//   * a weak pull to the centre stops disconnected nodes drifting away
//
// Seeded from the node index rather than Math.random, so the same graph always
// lays out the same way - a graph that reshuffles on every render is unreadable.

const ITERATIONS_BUDGET = 400_000; // caps O(n^2) work on large slices
const AREA_PER_NODE = 26_000;

/**
 * @param {{id: string, degree?: number}[]} nodes
 * @param {{source: string, target: string}[]} edges
 * @returns {Record<string, {x: number, y: number}>} position by node id
 */
export function layoutGraph(nodes, edges) {
  const count = nodes.length;
  if (count === 0) return {};

  const size = Math.max(600, Math.sqrt(count * AREA_PER_NODE));
  const centre = size / 2;

  // Seed on a spiral: deterministic, and already roughly evenly spread, which
  // means the simulation needs far fewer iterations to settle.
  const points = nodes.map((node, index) => {
    const angle = index * 2.399963; // golden angle, avoids seams
    const radius = (size / 2.4) * Math.sqrt((index + 0.5) / count);
    return {
      id: node.id,
      x: centre + radius * Math.cos(angle),
      y: centre + radius * Math.sin(angle),
      vx: 0,
      vy: 0,
      degree: node.degree ?? 1,
    };
  });

  const index = new Map(points.map((point) => [point.id, point]));
  const links = edges
    .map((edge) => ({ a: index.get(edge.source), b: index.get(edge.target) }))
    .filter((link) => link.a && link.b && link.a !== link.b);

  const iterations = Math.max(
    40,
    Math.min(220, Math.floor(ITERATIONS_BUDGET / Math.max(1, count * count))),
  );
  const idealLength = Math.max(120, size / Math.sqrt(count));

  for (let step = 0; step < iterations; step += 1) {
    // Cooling: large corrections early, fine adjustments later.
    const damping = 0.85 * (1 - step / iterations) + 0.05;

    for (let i = 0; i < count; i += 1) {
      const a = points[i];
      for (let j = i + 1; j < count; j += 1) {
        const b = points[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let distance = Math.hypot(dx, dy);
        if (distance < 0.01) {
          // Perfect overlap has no direction; nudge deterministically.
          dx = (i % 7) - 3;
          dy = (j % 7) - 3;
          distance = Math.hypot(dx, dy) || 1;
        }
        const force = (idealLength * idealLength) / (distance * distance);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
    }

    for (const { a, b } of links) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.hypot(dx, dy) || 1;
      const force = (distance - idealLength) * 0.08;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    for (const point of points) {
      point.vx += (centre - point.x) * 0.008;
      point.vy += (centre - point.y) * 0.008;
      // Clamp so a dense cluster cannot fling a node off-canvas.
      point.x += Math.max(-60, Math.min(60, point.vx * damping));
      point.y += Math.max(-60, Math.min(60, point.vy * damping));
      point.vx *= 0.5;
      point.vy *= 0.5;
    }
  }

  return Object.fromEntries(
    points.map((point) => [point.id, { x: Math.round(point.x), y: Math.round(point.y) }]),
  );
}
