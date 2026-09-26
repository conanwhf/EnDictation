const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
// Execute the actual polling implementation with controlled asynchronous responses.
const polling = html.slice(html.indexOf('const poll ='), html.indexOf("requeryBtn.addEventListener('click'"));
const upload = html.slice(html.indexOf('function uploadFile('), html.indexOf('// ---- 生成 TTS'));
const tts = html.slice(html.indexOf('function generateTts('), html.indexOf('// ---- 轮询'));

function harness() {
    const pending = [];
    const timers = [];
    const errors = [];
    const updates = [];
    const element = () => ({style: {}, value: 0, disabled: false, textContent: ''});
    const state = {
        fetch: () => new Promise((resolve, reject) => pending.push({resolve, reject})),
        setTimeout: (fn, delay) => { timers.push({fn, delay}); return timers.length; },
        clearTimeout: () => {},
        FormData: class { append() {} },
        POLL_INTERVAL_MS: 1000, NETWORK_RETRY_DELAYS_MS: [1000, 2000, 4000],
        requeryBtn: element(), statusMessage: element(),
        ttsControls: element(), generateTtsBtn: element(), selectFileBtn: element(),
        loading: element(), resultContainer: element(), progressBlock: element(),
        ttsEngineSelect: element(), ttsLanguageSelect: element(), voiceGenderSelect: element(), speedRange: element(),
        currentOcrData: [{text: 'Hello'}], uploadingLocked: false,
        hideAlerts() {}, onTaskFinishedUI() { errors.push('finished'); },
        showError: error => errors.push(error), handleTaskUpdate: task => updates.push(task),
    };
    vm.createContext(state);
    vm.runInContext(polling + upload + tts + '\nglobalThis.pollState = poll;', state);
    return {state, pending, timers, errors, updates};
}

const flush = () => new Promise(resolve => setImmediate(resolve));

for (const errorType of ['404', 'network', 'success']) {
    test(`late ${errorType} from A cannot affect polling B`, async () => {
        const {state, pending, errors, updates, timers} = harness();
        state.startPolling('A', 'tts');
        state.runPoll();
        state.startPolling('B', 'ocr');
        state.runPoll();
        const timerCount = timers.length;
        if (errorType === 'network') pending[0].reject(new Error('offline'));
        else pending[0].resolve({status: errorType === '404' ? 404 : 200, ok: true,
            json: async () => ({task_id: 'A', status: 'succeeded'})});
        await flush();
        assert.equal(state.pollState.active, true);
        assert.equal(state.pollState.taskId, 'B');
        assert.equal(state.pollState.retryCount, 0);
        assert.equal(timers.length, timerCount);
        assert.equal(errors.length, 0);
        assert.equal(updates.length, 0);
        pending[1].resolve({status: 200, ok: true, json: async () => ({task_id: 'B', status: 'succeeded'})});
        await flush();
        assert.equal(updates[0].task_id, 'B');
    });
}

test('upload invalidates the previous poll before its submission returns', async () => {
    const {state, pending, errors} = harness();
    state.startPolling('A', 'tts');
    state.runPoll();
    state.uploadFile({});
    pending[0].resolve({status: 404});
    await flush();
    assert.equal(errors.length, 0);
    assert.equal(state.selectFileBtn.disabled, true);
    pending[1].resolve({status: 202, json: async () => ({task_id: 'B'})});
    await flush();
    assert.equal(state.pollState.taskId, 'B');
    assert.equal(state.pollState.active, true);
});

for (const oldStatus of [202, 500]) {
    test(`late TTS submission ${oldStatus} cannot overwrite a newer upload`, async () => {
        const {state, pending, errors} = harness();
        state.generateTts();
        state.uploadFile({});
        pending[1].resolve({status: 202, json: async () => ({task_id: 'B'})});
        await flush();
        pending[0].resolve({status: oldStatus, json: async () => ({task_id: 'A', error: {message: 'old failure'}})});
        await flush();
        assert.equal(state.pollState.taskId, 'B');
        assert.equal(state.pollState.active, true);
        assert.equal(state.selectFileBtn.disabled, true);
        assert.equal(errors.length, 0);
    });
}

test('current 404 still stops polling and displays its error', async () => {
    const {state, pending, errors} = harness();
    state.startPolling('A', 'tts');
    state.runPoll();
    pending[0].resolve({status: 404});
    await flush();
    assert.equal(state.pollState.active, false);
    assert.equal(errors.length, 2);
});
