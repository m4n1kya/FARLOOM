/**
 * MaskedHeading — React Bits (Vanilla JS + CSS Port)
 * Features SVG text clip-path masking, mouse parallax, drift, spotlight, outlines, and rise reveal.
 */
(function() {
  'use strict';

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  const SPOT_MASK = `radial-gradient(circle var(--mh-spot-r, 160px) at var(--mh-spot-x, 50%) var(--mh-spot-y, 50%), ${Array.from(
    { length: 9 },
    (_, i) => {
      const t = i / 8;
      const u = clamp((t - 0.1) / 0.9, 0, 1);
      return `rgba(0, 0, 0, ${(u * u * u * (u * (u * 6 - 15) + 10)).toFixed(3)}) ${(t * 100).toFixed(1)}%`;
    }
  ).join(', ')})`;

  function createMaskedHeading(root, options = {}) {
    if (!root) return null;

    const text = options.text ?? 'FARLOOM';
    const mediaType = options.mediaType ?? 'image';
    const src = options.src ?? '/static/farloom-mask.svg';
    const poster = options.poster ?? '';
    const fillScale = options.fillScale ?? 1.3;
    const parallax = options.parallax ?? 34;
    const drift = options.drift ?? 10;
    const hoverZoom = options.hoverZoom ?? 0.06;
    const brightness = options.brightness ?? 1;
    const saturation = options.saturation ?? 1.15;
    const grayscale = options.grayscale ?? false;
    const tint = options.tint ?? '';
    const spotlight = options.spotlight ?? true;
    const spotlightRadius = options.spotlightRadius ?? 200;
    const outline = options.outline ?? true;
    const outlineColor = options.outlineColor ?? 'rgba(255, 255, 255, 0.45)';
    const outlineWidth = options.outlineWidth ?? 1;
    const reveal = options.reveal ?? 'rise';
    const duration = options.duration ?? 1.1;
    const stagger = options.stagger ?? 0.09;
    const trigger = options.trigger ?? 'view';
    const align = options.align ?? 'center';
    const weight = options.weight ?? 800;
    const tracking = options.tracking ?? -0.04;
    const lineHeight = options.lineHeight ?? 1.06;
    const textScale = options.textScale ?? 0.145;

    const clipId = 'mh-' + Math.random().toString(36).substring(2, 9);
    const words = String(text).split(/\s+/).filter(Boolean);
    const glyphs = words.flatMap((word, w) => Array.from(word).map(char => ({ char, w })));

    root.classList.add('masked-heading');
    if (options.className) {
      options.className.split(/\s+/).filter(Boolean).forEach(c => root.classList.add(c));
    }
    root.style.textAlign = align;
    root.style.fontWeight = weight;
    root.style.letterSpacing = `${tracking}em`;
    root.style.lineHeight = String(lineHeight);

    // Build DOM
    root.innerHTML = `
      <span class="masked-heading__measure">${words.map((word, i) => `<span class="masked-heading__word" data-word-idx="${i}">${word}<i class="masked-heading__baseline" data-base-idx="${i}"></i></span>`).join('')}</span>

      <svg class="masked-heading__defs" aria-hidden="true" focusable="false">
        <defs>
          <clipPath id="${clipId}" clipPathUnits="userSpaceOnUse">
            ${glyphs.map((g, i) => `<text data-glyph-idx="${i}">${g.char}</text>`).join('')}
          </clipPath>
        </defs>
      </svg>

      <span class="masked-heading__reveal">
        <span class="masked-heading__clip" style="clip-path: url(#${clipId}); --mh-spot-r: ${Math.max(1, spotlightRadius)}px;">
          <span class="masked-heading__media">
            ${mediaType === 'video' ? `
              <video class="masked-heading__source" src="${src}" ${poster ? `poster="${poster}"` : ''} autoplay muted loop playsinline></video>
            ` : `
              <img class="masked-heading__source" src="${src}" alt="" draggable="false" />
            `}
          </span>
          ${tint ? `<span class="masked-heading__tint" style="background:${tint}"></span>` : ''}
          ${spotlight ? `
            <span class="masked-heading__spot masked-heading__spot--tone"></span>
            <span class="masked-heading__spot masked-heading__spot--shade"></span>
          ` : ''}
        </span>
      </span>

      ${outline ? `
        <svg class="masked-heading__outline" aria-hidden="true" focusable="false" style="stroke: ${outlineColor}; stroke-width: ${Math.max(0, outlineWidth)}px;">
          ${glyphs.map((g, i) => `<text data-line-idx="${i}">${g.char}</text>`).join('')}
        </svg>
      ` : ''}
    `;

    const measure = root.querySelector('.masked-heading__measure');
    const clip = root.querySelector('.masked-heading__clip');
    const media = root.querySelector('.masked-heading__media');
    const revealLayer = root.querySelector('.masked-heading__reveal');
    const wordEls = Array.from(root.querySelectorAll('.masked-heading__word'));
    const baseEls = Array.from(root.querySelectorAll('.masked-heading__baseline'));
    const glyphEls = Array.from(root.querySelectorAll('.masked-heading__defs text'));
    const lineEls = Array.from(root.querySelectorAll('.masked-heading__outline text'));

    const spotTone = root.querySelector('.masked-heading__spot--tone');
    const spotShade = root.querySelector('.masked-heading__spot--shade');
    if (spotTone) {
      spotTone.style.maskImage = SPOT_MASK;
      spotTone.style.webkitMaskImage = SPOT_MASK;
    }
    if (spotShade) {
      spotShade.style.maskImage = SPOT_MASK;
      spotShade.style.webkitMaskImage = SPOT_MASK;
    }

    const offset = { x: 0, y: 0, tx: 0, ty: 0, zoom: 0, px: 0, py: 0, sx: 0, sy: 0, lit: false };
    const paint = { transform: '', filter: '', spot: '' };
    const states = glyphs.map(() => ({ x: 0, y: 0, s: 1 }));
    let boxes = [];
    let tween = null;

    const place = () => {
      if (!root || !media) return;
      const scale = fillScale * (1 + Math.max(0, offset.zoom));
      const maxX = Math.max(0, ((scale - 1) / 2) * root.clientWidth);
      const maxY = Math.max(0, ((scale - 1) / 2) * root.clientHeight);

      const transform = `translate3d(${clamp(offset.x, -maxX, maxX).toFixed(2)}px, ${clamp(offset.y, -maxY, maxY).toFixed(2)}px, 0) scale(${scale.toFixed(4)})`;
      const filter = `brightness(${brightness}) saturate(${saturation})${grayscale ? ' grayscale(1)' : ''}`;

      if (paint.transform !== transform) {
        paint.transform = transform;
        media.style.transform = transform;
      }
      if (paint.filter !== filter) {
        paint.filter = filter;
        media.style.filter = filter;
      }

      if (clip && spotlight) {
        const spot = `${offset.sx.toFixed(1)}px ${offset.sy.toFixed(1)}px`;
        if (paint.spot !== spot) {
          paint.spot = spot;
          clip.style.setProperty('--mh-spot-x', `${offset.sx.toFixed(1)}px`);
          clip.style.setProperty('--mh-spot-y', `${offset.sy.toFixed(1)}px`);
        }
      }
    };

    const renderGlyphs = () => {
      for (let i = 0; i < glyphEls.length; i += 1) {
        const glyph = glyphEls[i];
        if (!glyph) continue;
        const box = boxes[i];
        const st = states[i];
        if (!box || !st || (st.x === 0 && st.y === 0 && st.s === 1)) {
          if (glyph.hasAttribute('transform')) glyph.removeAttribute('transform');
          continue;
        }
        const s = Math.max(st.s, 0.0001);
        const tx = box.cx * (1 - s) + st.x * (box.cx - box.lc);
        const ty = box.cy * (1 - s) + st.y;
        glyph.setAttribute('transform', `translate(${tx.toFixed(2)} ${ty.toFixed(2)}) scale(${s.toFixed(4)})`);
      }
    };

    const sync = () => {
      if (!root || !measure) return;
      const targetSize = clamp(root.clientWidth * textScale, 28, 220);
      root.style.fontSize = `${targetSize.toFixed(1)}px`;

      const cs = window.getComputedStyle(measure);
      const fontPx = parseFloat(cs.fontSize) || targetSize;
      const rootRect = root.getBoundingClientRect();
      const ratio = root.offsetWidth ? rootRect.width / root.offsetWidth || 1 : 1;
      const range = document.createRange();
      const newBoxes = [];
      let index = 0;

      for (let w = 0; w < wordEls.length; w += 1) {
        const box = wordEls[w];
        const base = baseEls[w];
        const node = box ? Array.from(box.childNodes).find(n => n.nodeType === 3) : null;
        if (!box || !base || !node) {
          index += Array.from(box?.textContent ?? '').length;
          continue;
        }
        const boxRect = box.getBoundingClientRect();
        const baseline = base.offsetTop;
        let charOffset = 0;
        for (const char of Array.from(node.data)) {
          range.setStart(node, charOffset);
          range.setEnd(node, charOffset + char.length);
          charOffset += char.length;
          const r = range.getBoundingClientRect();
          const x = box.offsetLeft + (r.left - boxRect.left) / ratio;
          const width = r.width / ratio;
          newBoxes[index] = { x, y: baseline, w: width, cx: x + width / 2, cy: baseline - fontPx * 0.36, lc: 0 };
          index += 1;
        }
      }

      const lines = new Map();
      for (const box of newBoxes) {
        if (!box) continue;
        const line = lines.get(box.y) ?? { left: Infinity, right: -Infinity };
        line.left = Math.min(line.left, box.x);
        line.right = Math.max(line.right, box.x + box.w);
        lines.set(box.y, line);
      }
      for (const box of newBoxes) {
        if (!box) continue;
        const line = lines.get(box.y);
        box.lc = (line.left + line.right) / 2;
      }
      boxes = newBoxes;

      const write = el => {
        if (!el) return;
        el.style.fontFamily = cs.fontFamily;
        el.style.fontSize = cs.fontSize;
        el.style.fontWeight = cs.fontWeight;
        el.style.fontStyle = cs.fontStyle;
        el.style.letterSpacing = cs.letterSpacing;
      };

      for (let i = 0; i < boxes.length; i += 1) {
        const box = boxes[i];
        if (!box) continue;
        if (glyphEls[i]) {
          glyphEls[i].setAttribute('x', box.x.toFixed(2));
          glyphEls[i].setAttribute('y', String(box.y));
          write(glyphEls[i]);
        }
        if (lineEls[i]) {
          lineEls[i].setAttribute('x', box.x.toFixed(2));
          lineEls[i].setAttribute('y', String(box.y));
          write(lineEls[i]);
        }
      }
      renderGlyphs();
      place();
    };

    let alive = true;
    const ro = new ResizeObserver(() => sync());
    ro.observe(root);
    if (document.fonts?.ready) document.fonts.ready.then(() => alive && sync()).catch(() => {});

    const motionQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    let raf = 0;
    let last = 0;
    let clock = 0;
    let visible = true;
    let hovering = false;

    const frame = now => {
      raf = 0;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      const still = !!motionQuery?.matches;

      if (still) {
        offset.x = 0;
        offset.y = 0;
        offset.zoom = 0;
      } else {
        clock += dt;
        const dx = Math.sin(clock * 0.21) * drift;
        const dy = Math.cos(clock * 0.17) * drift * 0.6;
        const ease = 1 - Math.exp(-dt / 0.18);
        offset.x += (offset.tx + dx - offset.x) * ease;
        offset.y += (offset.ty + dy - offset.y) * ease;
        offset.zoom += ((hovering ? hoverZoom : 0) - offset.zoom) * (1 - Math.exp(-dt / 0.35));
      }

      if (spotlight) {
        const w = root.clientWidth;
        const h = root.clientHeight;
        let gx = w / 2;
        let gy = h / 2;
        if (hovering) {
          gx = offset.px;
          gy = offset.py;
        } else if (!still) {
          gx = w * (0.5 + 0.36 * Math.sin(clock * 0.37));
          gy = h * (0.5 + 0.22 * Math.sin(clock * 0.61 + 0.9));
        }
        if (!offset.lit || still) {
          offset.sx = gx;
          offset.sy = gy;
          offset.lit = true;
        } else {
          const follow = 1 - Math.exp(-dt / (hovering ? 0.12 : 0.6));
          offset.sx += (gx - offset.sx) * follow;
          offset.sy += (gy - offset.sy) * follow;
        }
      } else {
        offset.lit = false;
      }

      place();
      if (visible && !still && alive) raf = requestAnimationFrame(frame);
    };

    const wake = () => {
      if (raf || !alive) return;
      last = 0;
      raf = requestAnimationFrame(frame);
    };

    const onMove = e => {
      const r = root.getBoundingClientRect();
      const nx = ((e.clientX - r.left) / (r.width || 1)) * 2 - 1;
      const ny = ((e.clientY - r.top) / (r.height || 1)) * 2 - 1;
      offset.tx = clamp(nx, -1, 1) * -Math.max(0, parallax);
      offset.ty = clamp(ny, -1, 1) * -Math.max(0, parallax);
      offset.px = ((e.clientX - r.left) / (r.width || 1)) * root.clientWidth;
      offset.py = ((e.clientY - r.top) / (r.height || 1)) * root.clientHeight;
      if (spotlight) wake();
    };

    const onEnter = e => {
      hovering = e.pointerType !== 'touch';
      onMove(e);
      wake();
    };

    const onLeave = () => {
      hovering = false;
      offset.tx = 0;
      offset.ty = 0;
      wake();
    };

    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) wake();
    });
    io.observe(root);

    root.addEventListener('pointermove', onMove);
    root.addEventListener('pointerenter', onEnter);
    root.addEventListener('pointerleave', onLeave);

    // Reveal Animation
    const em = () => parseFloat(window.getComputedStyle(root).fontSize) || 100;
    const wordDelay = i => glyphs[i].w * stagger;

    const clearLayer = () => {
      if (window.gsap) window.gsap.set(revealLayer, { opacity: 1, scale: 1 });
      revealLayer.style.clipPath = '';
      revealLayer.style.filter = '';
    };

    const settle = () => {
      states.forEach(st => {
        st.x = 0;
        st.y = 0;
        st.s = 1;
      });
      renderGlyphs();
      clearLayer();
    };

    const rest = () => {
      const size = em();
      if (reveal === 'rise') {
        states.forEach(st => {
          st.y = size * 1.15;
        });
      }
      renderGlyphs();
    };

    const play = () => {
      if (!window.gsap) {
        settle();
        return;
      }
      tween?.kill();
      const size = em();
      if (reveal === 'rise') {
        clearLayer();
        tween = window.gsap.fromTo(
          states,
          { x: 0, y: size * 1.15, s: 1 },
          { y: 0, duration, stagger: wordDelay, ease: 'power4.out', overwrite: 'auto', onUpdate: renderGlyphs }
        );
        renderGlyphs();
      } else {
        settle();
      }
    };

    if (trigger === 'view') {
      settle();
      rest();
      const viewIo = new IntersectionObserver(
        entries => {
          if (entries.some(e => e.isIntersecting)) {
            play();
            viewIo.disconnect();
          }
        },
        { threshold: 0.15 }
      );
      viewIo.observe(root);
    } else {
      play();
    }

    sync();
    wake();

    return {
      sync,
      destroy: () => {
        alive = false;
        cancelAnimationFrame(raf);
        ro.disconnect();
        io.disconnect();
        root.removeEventListener('pointermove', onMove);
        root.removeEventListener('pointerenter', onEnter);
        root.removeEventListener('pointerleave', onLeave);
        tween?.kill();
      }
    };
  }

  window.createMaskedHeading = createMaskedHeading;
})();
