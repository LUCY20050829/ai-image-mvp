window.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#taskForm");
  const extraRequirement = document.querySelector("#extraRequirement");
  const poemSearch = document.querySelector("#poemSearch");
  const authorFilters = document.querySelector("#authorFilters");
  const poemList = document.querySelector("#poemList");
  const selectedPoems = document.querySelector("#selectedPoems");
  const selectedPoemCount = document.querySelector("#selectedPoemCount");
  const clearPoemsBtn = document.querySelector("#clearPoemsBtn");
  const selectAllPoemsBtn = document.querySelector("#selectAllPoemsBtn");
  const submitBtn = document.querySelector("#submitBtn");
  const resetBtn = document.querySelector("#resetBtn");
  const rerunBtn = document.querySelector("#rerunBtn");
  const stopBtn = document.querySelector("#stopBtn");
  const statusPill = document.querySelector("#statusPill");
  const statusSummary = document.querySelector("#statusSummary");
  const progressList = document.querySelector("#progressList");
  const resultGrid = document.querySelector("#resultGrid");
  const resultEmpty = document.querySelector("#resultEmpty");
  const downloadAllBtn = document.querySelector("#downloadAllBtn");

  let currentJobId = null;
  let eventSource = null;
  let running = false;
  let resultCount = 0;
  let hasSubmitted = false;

  const defaultState = {
    grade: "junior_high",
    hasPerson: true,
    hasPoem: false,
    aspectRatio: "16:9",
    imageCount: 4,
    extraRequirement: "",
    selectedPoemIds: [],
    selectedAuthorNames: [],
  };

  const sampleState = {
    grade: "junior_high",
    hasPerson: true,
    hasPoem: true,
    aspectRatio: "16:9",
    imageCount: 4,
    extraRequirement: "人物不要过于悲苦；整体色调更温暖；画面更开阔，适合作为章节头图。",
    selectedPoemIds: ["junior_high-001", "junior_high-002"],
    selectedAuthorNames: [],
  };

  const state = { ...defaultState };

  const gradeLabels = {
    lower_primary: "小低",
    middle_upper_primary: "小中小高",
    junior_high: "初中",
  };

  // 后端本轮不改造，所以这里临时映射到旧接口识别的学段值。
  const gradeSubmitMap = {
    lower_primary: "小学",
    middle_upper_primary: "小学",
    junior_high: "初中",
  };

  const poemData = window.POEM_DATA || {
    lower_primary: [],
    middle_upper_primary: [],
    junior_high: [],
  };

  const aspectLabels = {
    "1:1": "方图",
    "4:3": "横图",
    "3:4": "竖图",
    "16:9": "宽屏",
    "9:16": "长屏",
  };

  function escapeHtml(text) {
    return String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }


  function getCurrentPoems() {
    return poemData[state.grade] || [];
  }

  function getSelectedPoemItems() {
    const allPoems = Object.values(poemData).flat();
    return state.selectedPoemIds
      .map((id) => allPoems.find((poem) => poem.id === id))
      .filter(Boolean);
  }

  function renderSelectedPoems() {
    const items = getSelectedPoemItems();
    const totalImages = items.length * Number(state.imageCount || 0);
    selectedPoemCount.textContent = items.length
      ? `已选 ${items.length} 首，每首生成 ${state.imageCount} 张，一共 ${totalImages} 张`
      : "已选 0 首";
    selectedPoems.hidden = items.length === 0;
    selectedPoems.innerHTML = items.map((poem) => `<span>${escapeHtml(poem.title)}</span>`).join("");
  }

  function getCurrentAuthors() {
    return [...new Set(getCurrentPoems().map((poem) => poem.author).filter(Boolean))];
  }

  function getFilteredPoems() {
    const keyword = poemSearch.value.trim().toLowerCase();
    return getCurrentPoems().filter((poem) => {
      const text = `${poem.title} ${poem.author} ${poem.content}`.toLowerCase();
      const keywordMatched = !keyword || text.includes(keyword);
      const authorMatched = !state.selectedAuthorNames.length || state.selectedAuthorNames.includes(poem.author);
      return keywordMatched && authorMatched;
    });
  }

  function renderAuthorFilters() {
    const authors = getCurrentAuthors();
    authorFilters.innerHTML = authors.map((author) => {
      const active = state.selectedAuthorNames.includes(author) ? "is-active" : "";
      return `<button type="button" class="author-chip ${active}" data-author="${escapeHtml(author)}">${escapeHtml(author)}</button>`;
    }).join("");
  }

  function renderPoemList() {
    renderAuthorFilters();
    const rows = getFilteredPoems();

    if (!rows.length) {
      poemList.innerHTML = `<div class="poem-empty">没有找到匹配的诗歌</div>`;
      renderSelectedPoems();
      return;
    }

    poemList.innerHTML = rows.map((poem) => {
      const checked = state.selectedPoemIds.includes(poem.id) ? "checked" : "";
      return `
        <details class="poem-item">
          <summary>
            <label class="poem-check" onclick="event.stopPropagation()">
              <input type="checkbox" data-poem-id="${escapeHtml(poem.id)}" ${checked} />
              <span class="poem-title">${escapeHtml(poem.title)}</span>
              <span class="poem-author">${escapeHtml(poem.author)}</span>
            </label>
            <span class="poem-open">查看正文</span>
          </summary>
          <p>${escapeHtml(poem.content)}</p>
        </details>
      `;
    }).join("");
    renderSelectedPoems();
  }

  function togglePoem(poemId, checked) {
    if (checked && !state.selectedPoemIds.includes(poemId)) {
      state.selectedPoemIds.push(poemId);
    }
    if (!checked) {
      state.selectedPoemIds = state.selectedPoemIds.filter((id) => id !== poemId);
    }
    renderSelectedPoems();
  }

  function buildSelectedPoemText() {
    const items = getSelectedPoemItems();
    if (!items.length) return "";
    const poemLines = items.map((poem) => `《${poem.title}》${poem.author}：${poem.content}`);
    return `本次选择的诗歌：${poemLines.join("；")}`;
  }
  function toBool(value) {
    return value === true || value === "true";
  }

  function getPersonLabel() {
    return state.hasPerson ? "有人物" : "无人物";
  }

  function getPoemLabel() {
    return state.hasPoem ? "有诗词" : "无诗词";
  }

  function resolveImageType() {
    if (state.hasPoem) return "诗词意境";
    if (state.hasPerson) return "故事插图";
    return "场景背景";
  }

  function buildDefaultDescription() {
    const personText = state.hasPerson ? "包含清晰人物形象" : "不出现人物，以环境和物件表达主题";
    const poemText = state.hasPoem ? "画面中需要出现诗词文字或明显诗词内容" : "画面中不出现诗词文字";
    return `请生成适合${gradeLabels[state.grade]}教研使用的配图，${personText}，${poemText}，整体适合作为课堂或交互图书素材。`;
  }

  function setRunning(next) {
    running = next;
    submitBtn.disabled = next;
    resetBtn.disabled = next;
    if (rerunBtn) rerunBtn.disabled = next || !hasSubmitted;
    extraRequirement.disabled = next;
    stopBtn.disabled = !next;
    form.querySelectorAll("[data-choice-group]").forEach((el) => {
      el.disabled = next;
    });
  }

  function setStatus(text) {
    statusPill.textContent = text;
  }

  function setSummary(title, desc) {
    statusSummary.innerHTML = `<div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(desc)}</span></div>`;
  }

  function appendProgress(message, kind = "progress") {
    const row = document.createElement("div");
    row.className = `progress-item ${kind}`;
    row.innerHTML = `<span class="dot"></span><div>${escapeHtml(message)}</div>`;
    progressList.prepend(row);
  }

  function resetResults() {
    resultCount = 0;
    resultGrid.innerHTML = "";
    resultEmpty.hidden = false;
    downloadAllBtn.hidden = true;
    downloadAllBtn.dataset.href = "";
    downloadAllBtn.textContent = "下载全部结果";
    progressList.innerHTML = "";
  }

  function syncChoiceButtons(group, activeValue) {
    form.querySelectorAll(`[data-choice-group="${group}"]`).forEach((button) => {
      const isActive = button.dataset.value === String(activeValue);
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-checked", String(isActive));
    });
  }

  function applyState(nextState) {
    Object.assign(state, nextState);
    extraRequirement.value = state.extraRequirement || "";
    if (!Array.isArray(state.selectedPoemIds)) state.selectedPoemIds = [];
    if (!Array.isArray(state.selectedAuthorNames)) state.selectedAuthorNames = [];
    syncChoiceButtons("grade", state.grade);
    syncChoiceButtons("person", state.hasPerson);
    syncChoiceButtons("poem", state.hasPoem);
    syncChoiceButtons("aspect", state.aspectRatio);
    syncChoiceButtons("count", state.imageCount);
    if (poemSearch) poemSearch.value = "";
    renderPoemList();
  }

  function updateStateFromChoice(group, value) {
    if (group === "grade") {
      state.grade = value;
      state.selectedPoemIds = [];
      state.selectedAuthorNames = [];
    }
    if (group === "person") state.hasPerson = toBool(value);
    if (group === "poem") state.hasPoem = toBool(value);
    if (group === "aspect") state.aspectRatio = value;
    if (group === "count") state.imageCount = Number(value);
    syncChoiceButtons(group, group === "person" ? state.hasPerson : group === "poem" ? state.hasPoem : group === "aspect" ? state.aspectRatio : group === "count" ? state.imageCount : state.grade);
    if (group === "grade") renderPoemList();
    if (group === "count") renderSelectedPoems();
  }

  function buildPayload() {
    const trimmedExtra = extraRequirement.value.trim();
    state.extraRequirement = trimmedExtra;

    const selectedPoemText = buildSelectedPoemText();
    const finalDescription = [
      selectedPoemText,
      trimmedExtra ? `${u('\u8865\u5145\u8981\u6c42')}?${trimmedExtra}` : "",
    ].filter(Boolean).join("?") || buildDefaultDescription();

    return {
      grade_level: gradeSubmitMap[state.grade],
      grade_band: state.grade,
      image_type: resolveImageType(),
      aspect_ratio: state.aspectRatio,
      image_count: Number(state.imageCount),
      has_person: state.hasPerson,
      has_poem: state.hasPoem,
      unified_style: true,
      reserve_text_area: state.hasPoem,
      extra_requirement: trimmedExtra,
      selected_poems: getSelectedPoemItems().map((poem) => ({
        author: poem.author,
        title: poem.title,
        content: poem.content,
      })),
      user_description: finalDescription,
      save_outputs: false,
    };
  }

  function buildUiSnapshot() {
    return {
      grade: gradeLabels[state.grade],
      person: getPersonLabel(),
      poem: getPoemLabel(),
      aspect: state.aspectRatio,
      aspectText: aspectLabels[state.aspectRatio],
    };
  }


  function openImagePreview(image) {
    const overlay = document.createElement("div");
    overlay.className = "image-preview-overlay";
    overlay.innerHTML = `
      <div class="image-preview-dialog" role="dialog" aria-modal="true" aria-label="图片预览">
        <button type="button" class="image-preview-close" aria-label="关闭预览">×</button>
        <img src="${image.url}" alt="${escapeHtml(image.name)}" />
        <div class="image-preview-caption">
          <strong>${escapeHtml(image.name)}</strong>
          <a class="small-btn link-btn" href="${image.download_url || image.url}" download="${escapeHtml(image.name)}">下载这张</a>
        </div>
      </div>
    `;
    const close = () => overlay.remove();
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay || event.target.classList.contains("image-preview-close")) close();
    });
    document.addEventListener("keydown", function onKeydown(event) {
      if (event.key === "Escape") {
        close();
        document.removeEventListener("keydown", onKeydown);
      }
    });
    document.body.appendChild(overlay);
  }

  function addResultCard(image, payload, uiSnapshot) {
    resultEmpty.hidden = true;
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <button type="button" class="result-preview-btn" aria-label="预览 ${escapeHtml(image.name)}">
        <img src="${image.url}" alt="${escapeHtml(image.name)}" />
      </button>
      <div class="result-body">
        <strong>${escapeHtml(image.name)}</strong>
        <p>${escapeHtml(uiSnapshot.grade)} / ${escapeHtml(uiSnapshot.person)} / ${escapeHtml(uiSnapshot.poem)} / ${escapeHtml(uiSnapshot.aspect)}</p>
        <div class="card-actions">
          <button type="button" class="small-btn regen-btn">重新生成</button>
          <a class="small-btn link-btn" href="${image.download_url || image.url}" download="${escapeHtml(image.name)}">下载</a>
          <button type="button" class="small-btn fav-btn" data-fav="0">收藏</button>
        </div>
      </div>
    `;
    card.querySelector(".result-preview-btn").addEventListener("click", () => openImagePreview(image));
    resultGrid.appendChild(card);
    resultCount += 1;
  }

  async function stopJob() {
    if (!currentJobId) return;
    setStatus("停止中");
    setSummary("正在停止任务", "系统会在当前步骤结束后停止后续处理。");
    try {
      await fetch("/api/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: currentJobId }),
      });
    } catch (_) {}
  }

  async function downloadAll() {
    const href = downloadAllBtn.dataset.href;
    if (!href) return;
    downloadAllBtn.disabled = true;
    downloadAllBtn.textContent = "正在准备下载";
    try {
      const res = await fetch(href);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "ai-auto-image-result.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      downloadAllBtn.textContent = "已开始下载";
    } catch (error) {
      downloadAllBtn.textContent = `下载失败：${error}`;
    } finally {
      downloadAllBtn.disabled = false;
    }
  }


  async function handleSubmit(event) {
    event.preventDefault();
    if (running) return;

    const payload = buildPayload();
    const uiSnapshot = buildUiSnapshot();

    resetResults();
    hasSubmitted = true;
    setRunning(true);
    setStatus("启动中");
    setSummary("任务已提交", "系统正在根据结构化配置准备提示词与批量生成任务。");
    appendProgress(`已收到任务：${uiSnapshot.grade}，${uiSnapshot.person}，${uiSnapshot.poem}，${uiSnapshot.aspect}，选择 ${state.selectedPoemIds.length} 首诗歌，共 ${payload.image_count} 张。`);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "任务启动失败");

      currentJobId = data.job_id;
      eventSource = new EventSource(`/api/events?job=${data.job_id}`);
      setStatus("生成中");

      eventSource.onmessage = (msg) => {
        const eventData = JSON.parse(msg.data);
        if (eventData.kind === "progress") {
          appendProgress(eventData.message, "progress");
          setSummary("任务执行中", eventData.message || "系统正在处理中");
        }
        if (eventData.kind === "image" && eventData.image) {
          appendProgress(eventData.message, "image");
          addResultCard(eventData.image, payload, uiSnapshot);
          setSummary("已有结果生成", `当前已生成 ${resultCount} 张图片。`);
        }
        if (eventData.kind === "done") {
          appendProgress(eventData.message, eventData.partial ? "warn" : "done");
          setStatus(eventData.partial ? "部分完成" : "已完成");
          setSummary(eventData.title || (eventData.partial ? "部分完成" : "全部成功"), eventData.message || "本次任务已处理完成。");
          setRunning(false);
          currentJobId = null;
          if (eventSource) eventSource.close();
          eventSource = null;
          if (eventData.download) {
            downloadAllBtn.hidden = false;
            downloadAllBtn.dataset.href = eventData.download;
          }
        }
        if (eventData.kind === "cancelled") {
          appendProgress(eventData.message, "warn");
          setStatus("已停止");
          setSummary(eventData.title || "任务已停止", eventData.message || "后续步骤已取消，已生成内容仍会保留。");
          setRunning(false);
          currentJobId = null;
          if (eventData.download) {
            downloadAllBtn.hidden = false;
            downloadAllBtn.dataset.href = eventData.download;
          }
        }
        if (eventData.kind === "error") {
          appendProgress(eventData.message, "error");
          setStatus(eventData.partial ? "部分完成" : "失败");
          setSummary(eventData.title || "任务失败", eventData.message || "请稍后重试或缩小任务范围。");
          setRunning(false);
          currentJobId = null;
          if (eventData.download) {
            downloadAllBtn.hidden = false;
            downloadAllBtn.dataset.href = eventData.download;
          }
        }
        if (eventData.kind === "close") {
          if (eventSource) eventSource.close();
          eventSource = null;
          currentJobId = null;
          setRunning(false);
        }
      };

      eventSource.onerror = () => {
        appendProgress("进度连接中断，请查看后端运行日志。", "error");
        setStatus("连接中断");
        setSummary("进度连接已中断", "如果任务仍在运行，可以稍后刷新页面查看。");
        if (eventSource) eventSource.close();
        eventSource = null;
        currentJobId = null;
        setRunning(false);
      };
    } catch (error) {
      appendProgress(`任务启动失败：${error}`, "error");
      setStatus("失败");
      setSummary("启动失败", "请检查后端服务状态，或稍后缩小任务数量重试。");
      setRunning(false);
      currentJobId = null;
    }
  }

  form.addEventListener("click", (event) => {
    const choice = event.target.closest("[data-choice-group]");
    if (!choice || running) return;
    updateStateFromChoice(choice.dataset.choiceGroup, choice.dataset.value);
  });

  poemSearch.addEventListener("input", renderPoemList);
  authorFilters.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-author]");
    if (!chip || running) return;
    const author = chip.dataset.author;
    if (state.selectedAuthorNames.includes(author)) {
      state.selectedAuthorNames = state.selectedAuthorNames.filter((item) => item !== author);
    } else {
      state.selectedAuthorNames.push(author);
    }
    renderPoemList();
  });
  selectAllPoemsBtn.addEventListener("click", () => {
    if (running) return;
    getFilteredPoems().forEach((poem) => {
      if (!state.selectedPoemIds.includes(poem.id)) state.selectedPoemIds.push(poem.id);
    });
    renderPoemList();
  });
  poemList.addEventListener("change", (event) => {
    const checkbox = event.target.closest("[data-poem-id]");
    if (!checkbox || running) return;
    togglePoem(checkbox.dataset.poemId, checkbox.checked);
  });
  clearPoemsBtn.addEventListener("click", () => {
    if (running) return;
    state.selectedPoemIds = [];
    renderPoemList();
  });

  function rerunCurrentTask() {
    if (running) return;
    resetResults();
    setStatus("待命");
    setSummary("准备重新生成", "将使用当前任务配置重新提交一轮生成。");
    if (form.requestSubmit) {
      form.requestSubmit();
    } else {
      submitBtn.click();
    }
  }

  form.addEventListener("submit", handleSubmit);
  resetBtn.addEventListener("click", () => {
    if (running) return;
    applyState(defaultState);
    setStatus("待命");
    setSummary("等待提交", "选择任务配置后点击“开始生成”。");
    resetResults();
  });
  if (rerunBtn) rerunBtn.addEventListener("click", rerunCurrentTask);
  stopBtn.addEventListener("click", stopJob);
  downloadAllBtn.addEventListener("click", downloadAll);
  resultGrid.addEventListener("click", (event) => {
    const fav = event.target.closest(".fav-btn");
    if (fav) {
      const active = fav.dataset.fav === "1";
      fav.dataset.fav = active ? "0" : "1";
      fav.textContent = active ? "收藏" : "已收藏";
    }
    const regen = event.target.closest(".regen-btn");
    if (regen) {
      window.alert("MVP 版本先保留按钮占位，下一步可以接入单张重生成能力。");
    }
  });

  applyState(defaultState);
});


