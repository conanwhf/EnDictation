document.addEventListener('DOMContentLoaded', () => {
    const el = name => document.getElementById(`config-${name}`);
    const dialog = el('dialog');
    if (!dialog) return;
    let draft = null;
    const clone = data => JSON.parse(JSON.stringify(data));

    function fail(error) {
        el('error').textContent = error.message;
        el('error').hidden = false;
        el('error').scrollIntoView({block: 'nearest'});
    }
    async function request(options) {
        const response = await fetch('/config', {cache: 'no-store', ...options});
        const body = await response.json();
        if (!response.ok) throw new Error(body.error?.message || '配置请求失败');
        return body;
    }
    function account() {
        const info = draft.secrets.google_cloud_tts_credentials_json;
        el('account').textContent = info ? `已配置：${info.client_email || '服务账号'}` : '未配置服务账号';
        el('google-clear').disabled = !info;
    }
    function display(data) {
        if (!data || data.version !== 1 || !data.secrets || !Array.isArray(data.tts_engines) ||
            !data.tts_engines.length || !Array.isArray(data.tts_languages) || !data.tts_models || !data.tts_voice_matrix) {
            throw new Error('配置文件不完整或版本不支持');
        }
        // Check references before replacing the editor's current draft.
        for (const engine of data.tts_engines) {
            const languages = data.tts_voice_matrix[engine.id];
            if (!languages || !Object.keys(languages).length) throw new Error('每个引擎至少需要一种语言');
            for (const [id, voices] of Object.entries(languages)) {
                if (!data.tts_languages.some(language => language.id === id) ||
                    !voices || !Object.keys(voices).length ||
                    Object.values(voices).some(key => !data.tts_models[key])) {
                    throw new Error('语言或音色引用不完整');
                }
            }
        }
        if (Object.hasOwn(data, 'preferred_tts') && (!data.preferred_tts ||
            !data.tts_voice_matrix[data.preferred_tts.engine] ||
            !data.tts_voice_matrix[data.preferred_tts.engine][data.preferred_tts.language])) {
            throw new Error('首选引擎和语言组合不存在');
        }
        draft = clone(data);
        if (!draft.preferred_tts) {
            const engine = draft.tts_voice_matrix.azure ? 'azure' : draft.tts_engines[0].id;
            const languages = draft.tts_voice_matrix[engine];
            draft.preferred_tts = {engine, language: languages['sg-en'] ? 'sg-en' : Object.keys(languages)[0]};
        }
        el('gemini').value = data.secrets.google_api_key || '';
        el('azure').value = data.secrets.azure_speech_key || '';
        el('ocr').value = data.ocr_model || '';
        const previous = el('engine').value;
        el('engine').replaceChildren();
        data.tts_engines.forEach(engine => el('engine').add(new Option(engine.label, engine.id)));
        if (data.tts_engines.some(engine => engine.id === previous)) el('engine').value = previous;
        account();
        renderLanguages();
        el('fields').disabled = false;
        el('error').hidden = true;
    }
    function collect() {
        const data = clone(draft);
        data.ocr_model = el('ocr').value.trim();
        data.secrets.google_api_key = el('gemini').value.trim() || null;
        data.secrets.azure_speech_key = el('azure').value.trim() || null;
        return data;
    }
    function uniqueId(object, prefix) {
        let number = 1;
        while (Object.hasOwn(object, `${prefix}-${number}`)) number++;
        return `${prefix}-${number}`;
    }
    function input(label, value, onInput, placeholder = '') {
        const wrapper = document.createElement('label');
        wrapper.textContent = label;
        const field = document.createElement('input');
        field.className = 'form-control mt-1';
        field.value = value || '';
        field.placeholder = placeholder;
        field.required = true;
        field.addEventListener('input', () => onInput(field.value.trim()));
        wrapper.append(field);
        return wrapper;
    }
    function setVoice(engine, language, gender, field, value) {
        const map = draft.tts_voice_matrix[engine][language];
        let key = map[gender];
        const targets = draft.tts_models[key].type === 'gtts' ? Object.keys(map) : [gender];
        const references = Object.values(draft.tts_voice_matrix).flatMap(languages =>
            Object.values(languages).flatMap(voices => Object.values(voices))).filter(id => id === key).length;
        // Editing one language must not silently change another language sharing its voice.
        if (references > targets.filter(g => map[g] === key).length) {
            const replacement = uniqueId(draft.tts_models, 'custom-voice');
            draft.tts_models[replacement] = clone(draft.tts_models[key]);
            targets.forEach(g => { map[g] = replacement; });
            key = replacement;
        }
        draft.tts_models[key][field] = value;
    }
    function renderLanguages() {
        const engine = el('engine').value;
        const mappings = draft.tts_voice_matrix[engine];
        el('languages').replaceChildren();
        const preferred = draft.preferred_tts;
        const preferredEngine = draft.tts_engines.find(item => item.id === preferred.engine);
        const preferredLanguage = draft.tts_languages.find(item => item.id === preferred.language);
        el('preferred').textContent = `当前首选：${preferredEngine.label} · ${preferredLanguage.label}`;
        for (const [id, voices] of Object.entries(mappings)) {
            const language = draft.tts_languages.find(item => item.id === id);
            const row = document.createElement('div');
            row.className = 'config-language';
            const heading = document.createElement('div');
            heading.className = 'd-flex gap-2 align-items-end';
            const name = input('语言名称', language.label, value => { language.label = value; });
            name.className = 'flex-grow-1';
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'btn btn-outline-danger';
            remove.title = `移除${language.label}`;
            remove.setAttribute('aria-label', remove.title);
            remove.innerHTML = '<i class="bi bi-trash" aria-hidden="true"></i>';
            remove.addEventListener('click', () => {
                if (draft.preferred_tts.engine === engine && draft.preferred_tts.language === id) {
                    fail(new Error('请先将另一种语言设为首选，再移除此项'));
                    return;
                }
                if (Object.keys(mappings).length === 1) {
                    fail(new Error('每个引擎至少保留一种语言'));
                    return;
                }
                delete mappings[id];
                const allMappings = Object.values(draft.tts_voice_matrix);
                if (!allMappings.some(languages => Object.hasOwn(languages, id))) {
                    draft.tts_languages = draft.tts_languages.filter(item => item.id !== id);
                }
                const referenced = new Set(allMappings.flatMap(languages =>
                    Object.values(languages).flatMap(items => Object.values(items))));
                Object.values(voices).forEach(key => {
                    if (!referenced.has(key)) delete draft.tts_models[key];
                });
                renderLanguages();
            });
            heading.append(name, remove);
            const preferredLabel = document.createElement('label');
            preferredLabel.className = 'd-flex align-items-center gap-2 mt-2';
            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'preferred-tts';
            radio.value = id;
            radio.className = 'form-check-input m-0';
            radio.checked = preferred.engine === engine && preferred.language === id;
            radio.addEventListener('change', () => {
                draft.preferred_tts = {engine, language: id};
                el('error').hidden = true;
                renderLanguages();
            });
            preferredLabel.append(radio, document.createTextNode('设为首选'));
            const fields = document.createElement('div');
            fields.className = 'config-voice-fields';
            const gender = Object.keys(voices)[0];
            const model = draft.tts_models[voices[gender]];
            const edit = (g, field) => value => setVoice(engine, id, g, field, value);
            if (model.type === 'gtts') {
                fields.append(input('语言代码', model.lang, edit(gender, 'lang'), '例如 en、zh、fr'),
                    input('口音域名', model.tld, edit(gender, 'tld'), '例如 com、co.uk'));
            } else {
                if (model.type === 'google-cloud-tts') {
                    fields.append(input('语言代码', model.language_code, value => {
                        Object.keys(voices).forEach(g => setVoice(engine, id, g, 'language_code', value));
                    }, '例如 en-US'));
                }
                for (const g of Object.keys(voices)) {
                    fields.append(input(g === 'male' ? '男声音色' : '女声音色',
                        draft.tts_models[voices[g]].voice_name, edit(g, 'voice_name')));
                }
            }
            row.append(heading, preferredLabel, fields);
            el('languages').append(row);
        }
    }
    el('engine').addEventListener('change', renderLanguages);
    el('add-language').addEventListener('click', () => {
        const engine = el('engine').value;
        const mapping = draft.tts_voice_matrix[engine];
        const first = Object.values(Object.values(mapping)[0])[0];
        const type = draft.tts_models[first].type;
        const id = uniqueId(Object.fromEntries(draft.tts_languages.map(l => [l.id, true])), 'custom-language');
        draft.tts_languages.push({id, label: ''});
        mapping[id] = {};
        for (const gender of ['male', 'female']) {
            if (type === 'gtts' && gender === 'female') {
                mapping[id].female = mapping[id].male;
                continue;
            }
            const key = uniqueId(draft.tts_models, 'custom-voice');
            draft.tts_models[key] = {label: '自定义音色', type,
                ...(type === 'gtts' ? {lang: '', tld: 'com'} : {voice_name: ''}),
                ...(type === 'google-cloud-tts' ? {language_code: ''} : {})};
            mapping[id][gender] = key;
        }
        renderLanguages();
        el('languages').lastElementChild.querySelector('input').focus();
    });
    el('btn').addEventListener('click', async () => {
        dialog.showModal();
        el('fields').disabled = true;
        el('error').hidden = true;
        try { display(await request()); } catch (error) { fail(error); }
    });
    ['close', 'cancel'].forEach(name => el(name).addEventListener('click', () => dialog.close()));
    dialog.addEventListener('close', () => {
        ['gemini', 'azure'].forEach(name => { el(name).type = 'password'; });
        document.querySelectorAll('.config-reveal').forEach(button => {
            button.querySelector('i').className = 'bi bi-eye';
            button.title = '显示密钥';
            button.setAttribute('aria-pressed', 'false');
        });
    });
    document.querySelectorAll('.config-reveal').forEach(button => button.addEventListener('click', () => {
        const field = document.getElementById(button.dataset.target);
        field.type = field.type === 'password' ? 'text' : 'password';
        button.querySelector('i').className = field.type === 'password' ? 'bi bi-eye' : 'bi bi-eye-slash';
        button.title = field.type === 'password' ? '显示密钥' : '隐藏密钥';
        button.setAttribute('aria-pressed', String(field.type === 'text'));
    }));
    el('import').addEventListener('click', () => el('file').click());
    el('google-import').addEventListener('click', () => el('google-file').click());
    async function readFile(target) {
        const file = target.files[0];
        target.value = '';
        if (!file) return null;
        try { return JSON.parse(await file.text()); }
        catch { throw new Error('文件不是有效的 JSON 配置'); }
    }
    el('file').addEventListener('change', async event => {
        try { const data = await readFile(event.target); if (data) display(data); }
        catch (error) { fail(error); }
    });
    el('google-file').addEventListener('change', async event => {
        try {
            const info = await readFile(event.target);
            if (!info) return;
            if (info.type !== 'service_account') throw new Error('请选择 Google Cloud 服务账号 JSON 文件');
            draft.secrets.google_cloud_tts_credentials_json = info;
            account();
            el('error').hidden = true;
        } catch (error) { fail(error); }
    });
    el('google-clear').addEventListener('click', () => {
        draft.secrets.google_cloud_tts_credentials_json = null;
        account();
    });
    el('export').addEventListener('click', () => {
        const url = URL.createObjectURL(new Blob([JSON.stringify(collect(), null, 2)], {type: 'application/json'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = 'endictation-config.json';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    el('form').addEventListener('submit', async event => {
        event.preventDefault();
        el('fields').disabled = true;
        try {
            const options = await request({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(collect())});
            document.dispatchEvent(new CustomEvent('configuration-saved', {detail: options}));
            dialog.close();
        } catch (error) { fail(error); }
        finally { el('fields').disabled = false; }
    });
});
