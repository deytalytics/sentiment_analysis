(() => {
  const MAX_FILES = 10;
  const ALLOWED = /\.(png|jpe?g|webp|gif|pdf|txt)$/i;
  const files = [];

  const $ = (s) => document.querySelector(s);
  const drop = $("#drop");
  const input = $("#file-input");
  const list = $("#file-list");
  const form = $("#form");
  const statusEl = $("#status");
  const submitBtn = $("#submit");
  const results = $("#results");

  // Safe DOM builder: everything is inserted as text, never as HTML.
  function el(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") node.className = v;
      else node.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", isError);
  }

  // ---------- File intake ----------
  function addFiles(incoming) {
    for (const f of incoming) {
      if (files.length >= MAX_FILES) {
        setStatus(`You can add up to ${MAX_FILES} files.`, true);
        break;
      }
      const looksOk = f.type.startsWith("image/") || ALLOWED.test(f.name);
      if (!looksOk) {
        setStatus(`"${f.name}" isn't a supported type. Use PNG, JPG, WEBP, GIF, PDF or TXT.`, true);
        continue;
      }
      files.push(f);
      setStatus("");
    }
    renderFiles();
  }

  function renderFiles() {
    list.replaceChildren();
    files.forEach((f, i) => {
      const thumb = f.type.startsWith("image/")
        ? el("img", { class: "thumb", alt: "", src: URL.createObjectURL(f) })
        : el("span", { class: "thumb" }, f.name.split(".").pop().toUpperCase());
      const remove = el("button", { type: "button", "aria-label": `Remove ${f.name}` }, "Remove");
      remove.addEventListener("click", () => {
        files.splice(i, 1);
        renderFiles();
      });
      list.append(el("li", {}, thumb, el("span", { class: "name" }, f.name), remove));
    });
  }

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    addFiles(Array.from(input.files));
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((t) =>
    drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); })
  );
  ["dragleave", "drop"].forEach((t) =>
    drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); })
  );
  drop.addEventListener("drop", (e) => addFiles(Array.from(e.dataTransfer.files)));

  // Paste screenshots straight from the clipboard.
  document.addEventListener("paste", (e) => {
    const imgs = Array.from(e.clipboardData?.files || []).filter((f) => f.type.startsWith("image/"));
    if (imgs.length) addFiles(imgs);
  });

  // ---------- Loading state ----------
  const btnLabel = submitBtn.textContent;

  function setLoading(on) {
    submitBtn.disabled = on;
    submitBtn.setAttribute("aria-busy", String(on));
    submitBtn.replaceChildren(
      ...(on ? [el("span", { class: "spinner", "aria-hidden": "true" }), "Analysing\u2026"] : [btnLabel])
    );
  }

  function showLoadingCard() {
    results.replaceChildren(
      el("div", { class: "loading-card" },
        el("span", { class: "spinner large", "aria-hidden": "true" }),
        el("div", {},
          el("strong", {}, "Analysing your conversation"),
          el("p", {}, "Gemini is reading your messages. This usually takes 20 to 60 seconds.")
        )
      )
    );
    // On narrow screens the results sit below the form, so bring them into view.
    if (window.matchMedia("(max-width: 880px)").matches) {
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // ---------- Submit ----------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#text").value.trim();
    if (!files.length && !text) {
      setStatus("Add a screenshot, a file, or some pasted text first.", true);
      return;
    }

    const body = new FormData();
    files.forEach((f) => body.append("files", f, f.name));
    body.append("text", text);
    body.append("perspective", $("#perspective").value.trim());
    body.append("stage", $("#stage").value);

    const previousResults = Array.from(results.childNodes); // restored if the request fails
    setLoading(true);
    showLoadingCard();
    setStatus("");

    try {
      const res = await fetch("/analyze", { method: "POST", body });
      let data = {};
      try { data = await res.json(); } catch { /* non-JSON error page */ }
      if (!res.ok) throw new Error(data.error || "The analysis failed. Please try again.");
      render(data);
      setStatus("Analysis ready.");
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      results.replaceChildren(...previousResults);
      setStatus(err.message, true);
    } finally {
      setLoading(false);
    }
  });

  // ---------- Rendering ----------
  const SEV_ORDER = { high: 0, medium: 1, low: 2 };
  const SEV_LABEL = { high: "High concern", medium: "Medium concern", low: "Low concern" };
  const WHO_LABEL = { you: "From you", them: "From them", both: "Both of you" };

  function vennSvg(score) {
    const s = Math.max(0, Math.min(100, Number(score) || 0));
    const half = (150 - s * 1.4) / 2; // distance of each circle from centre
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 320 190");
    svg.setAttribute("class", "venn");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Two circles overlapping, representing a compatibility score of ${s} out of 100`);
    const mk = (cls, name, dir) => {
      const c = document.createElementNS(svg.namespaceURI, "circle");
      c.setAttribute("class", cls);
      c.setAttribute("cx", "160");
      c.setAttribute("cy", "85");
      c.setAttribute("r", "64");
      c.style.transform = `translateX(${dir * 75}px)`; // start apart
      requestAnimationFrame(() => requestAnimationFrame(() => {
        c.style.transform = `translateX(${dir * half}px)`;
      }));
      return c;
    };
    const label = (x, t) => {
      const tx = document.createElementNS(svg.namespaceURI, "text");
      tx.setAttribute("x", x);
      tx.setAttribute("y", "182");
      tx.setAttribute("text-anchor", "middle");
      tx.textContent = t;
      return tx;
    };
    svg.append(mk("v-you", "You", -1), mk("v-them", "Them", 1), label(70, "You"), label(250, "Them"));
    return svg;
  }

  function section(title, ...kids) {
    return el("section", {}, title && el("h2", {}, title), ...kids);
  }

  function render(r) {
    results.replaceChildren();

    if (r.safety_concern) {
      results.append(el("div", { class: "notice safety", role: "alert" },
        el("strong", {}, "Your safety comes first"),
        "Parts of this conversation show signs of threats, coercion, intimidation or a possible scam. " +
        "If you feel unsafe, contact local emergency services or a domestic abuse helpline, and avoid sending money or personal details."
      ));
    }
    for (const n of r.server_notes || []) results.append(el("div", { class: "notice" }, n));

    if (!r.analysable) {
      results.append(section("We couldn't analyse this",
        el("p", {}, r.extraction_notes || "The input didn't look like a readable conversation. Try a clearer screenshot or paste the text.")));
      return;
    }

    // Compatibility
    const c = r.compatibility;
    if (c) {
      results.append(section(null,
        el("div", { class: "score" },
          vennSvg(c.score),
          el("div", {},
            el("div", { class: "score-num" }, String(c.score), el("small", {}, " / 100")),
            el("div", { class: "score-label" }, c.label),
            el("div", { class: "score-conf" }, `Confidence: ${c.confidence}`),
            el("p", { class: "rationale" }, c.rationale)
          )
        ),
        r.summary && el("p", { class: "summary" }, r.summary),
        el("ul", { class: "breakdown" },
          (c.breakdown || []).map((d) =>
            el("li", {},
              el("div", { class: "dim-head" }, el("span", {}, d.dimension), el("span", {}, String(d.score))),
              el("div", { class: "bar", role: "img", "aria-label": `${d.dimension}: ${d.score} out of 100` },
                el("i", { style: `width:${Math.max(0, Math.min(100, d.score))}%` })),
              el("p", { class: "dim-note" }, d.note)
            )
          )
        )
      ));
    } else if (r.summary) {
      results.append(section("Summary", el("p", {}, r.summary)));
    }

    // Warnings
    const warnings = [...(r.warnings || [])].sort((a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity]);
    results.append(section("Warnings",
      warnings.length
        ? el("ul", { class: "cards" }, warnings.map((w) =>
            el("li", { class: `warning ${w.severity}` },
              el("div", { class: "tags" },
                el("span", { class: `sev ${w.severity}` }, SEV_LABEL[w.severity] || w.severity),
                el("h3", {}, w.category),
                el("span", { class: "who" }, WHO_LABEL[w.who] || "")
              ),
              w.quote && el("blockquote", {}, `\u201C${w.quote}\u201D`),
              el("p", {}, w.explanation)
            )))
        : el("p", { class: "none" }, "No warning signs found in this excerpt.")
    ));

    // Positives
    if ((r.positives || []).length) {
      results.append(section("What's working",
        el("ul", { class: "cards" }, r.positives.map((p) =>
          el("li", { class: "positive" },
            p.quote && el("blockquote", {}, `\u201C${p.quote}\u201D`),
            el("p", {}, p.explanation)
          )))
      ));
    }

    // Suggestions
    if ((r.suggestions || []).length) {
      results.append(section("What you could do next",
        el("ul", { class: "next" }, r.suggestions.map((s) => el("li", {}, s)))));
    }

    // Transcript for verification
    if ((r.transcript || []).length) {
      results.append(el("section", {},
        el("details", { class: "transcript" },
          el("summary", {}, `What we read from your files (${r.transcript.length} messages)`),
          el("p", { class: "hint" },
            "Check this matches your chat. If it picked the wrong sender, say which messages are yours and run it again."),
          r.extraction_notes && el("p", { class: "hint" }, `Note: ${r.extraction_notes}`),
          el("div", { class: "chat" }, r.transcript.map((m) => el("div", { class: `bubble ${m.speaker}` }, m.text)))
        )
      ));
    }

    results.append(el("p", { class: "disclaimer" },
      "This is an AI reading of the messages you shared, not professional advice or a prediction about the relationship. " +
      "It can misread tone, sarcasm and context."));
  }
})();
