<script lang="ts" setup>
import { type DataTableColumns, NDataTable, NInputNumber, NSpin } from 'naive-ui';
import { apiManualImport } from '@/api/manualImport';
import type {
  FileMapping,
  FolderFileMapping,
  FolderImportPreview,
  ImportApplyResult,
  ImportCandidate,
  ImportPreview,
} from '#/manualImport';

const show = defineModel('show', { default: false });

const { t } = useMyI18n();
const message = useMessage();

type Step = 'pick' | 'form' | 'preview' | 'result';
type Mode = 'torrent' | 'folder' | 'upload';
type AnyMapping = FileMapping | FolderFileMapping;
const step = ref<Step>('pick');
const mode = ref<Mode>('torrent');
// Once picked (from the candidate list or via upload), both behave
// identically from here on -- same preview/apply endpoints, keyed by hash.
const usesTorrentFlow = computed(
  () => mode.value === 'torrent' || mode.value === 'upload'
);

const candidates = ref<ImportCandidate[]>([]);
const selected = ref<ImportCandidate | null>(null);
const folderPath = ref('');
const uploadFile = ref<File | null>(null);
const officialTitle = ref('');
const season = ref(1);
const preview = ref<ImportPreview | FolderImportPreview | null>(null);
const results = ref<ImportApplyResult[]>([]);

const loading = reactive({
  candidates: false,
  upload: false,
  preview: false,
  apply: false,
});

async function loadCandidates() {
  loading.candidates = true;
  try {
    candidates.value = await apiManualImport.getCandidates();
  } catch (e) {
    message.error(t('manual_import.load_failed'));
  } finally {
    loading.candidates = false;
  }
}

function pick(candidate: ImportCandidate) {
  selected.value = candidate;
  officialTitle.value = candidate.name;
  step.value = 'form';
}

function pickFolder() {
  const path = folderPath.value.trim();
  if (!path) {
    message.error(t('manual_import.folder_path_required'));
    return;
  }
  officialTitle.value = path.split(/[/\\]/).filter(Boolean).pop() ?? '';
  step.value = 'form';
}

function onFileSelected(e: Event) {
  const input = e.target as HTMLInputElement;
  uploadFile.value = input.files?.[0] ?? null;
}

async function pickUpload() {
  if (!uploadFile.value) {
    message.error(t('manual_import.upload_file_required'));
    return;
  }
  loading.upload = true;
  try {
    const candidate = await apiManualImport.uploadTorrent(uploadFile.value);
    selected.value = candidate;
    officialTitle.value = candidate.name;
    step.value = 'form';
  } catch (e) {
    message.error(t('manual_import.upload_failed'));
  } finally {
    loading.upload = false;
  }
}

async function runPreview() {
  if (usesTorrentFlow.value && !selected.value) return;
  if (!officialTitle.value.trim()) {
    message.error(t('manual_import.title_required'));
    return;
  }
  loading.preview = true;
  try {
    preview.value = usesTorrentFlow.value
      ? await apiManualImport.preview(
          selected.value!.hash,
          officialTitle.value.trim(),
          season.value
        )
      : await apiManualImport.previewFolder(
          folderPath.value.trim(),
          officialTitle.value.trim(),
          season.value
        );
    step.value = 'preview';
  } catch (e) {
    message.error(t('manual_import.preview_failed'));
  } finally {
    loading.preview = false;
  }
}

async function confirmApply() {
  if (!preview.value) return;
  if (usesTorrentFlow.value && !selected.value) return;
  loading.apply = true;
  try {
    results.value = usesTorrentFlow.value
      ? await apiManualImport.apply(
          selected.value!.hash,
          preview.value.target_folder,
          preview.value.mappings as FileMapping[]
        )
      : await apiManualImport.applyFolder(
          preview.value.mappings as FolderFileMapping[]
        );
    step.value = 'result';
  } catch (e) {
    message.error(t('manual_import.apply_failed'));
  } finally {
    loading.apply = false;
  }
}

function reset() {
  step.value = 'pick';
  mode.value = 'torrent';
  selected.value = null;
  folderPath.value = '';
  uploadFile.value = null;
  officialTitle.value = '';
  season.value = 1;
  preview.value = null;
  results.value = [];
}

function goBack() {
  if (step.value === 'form') step.value = 'pick';
  else if (step.value === 'preview') step.value = 'form';
}

function close() {
  show.value = false;
}

watch(show, (val) => {
  if (val) {
    reset();
    loadCandidates();
  }
});

function formatSize(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

const candidateColumns: DataTableColumns<ImportCandidate> = [
  { title: () => t('manual_import.torrent_name'), key: 'name', ellipsis: { tooltip: true } },
  {
    title: () => t('manual_import.size'),
    key: 'size',
    width: 100,
    render: (row) => formatSize(row.size),
  },
  { title: () => t('manual_import.state'), key: 'state', width: 120 },
  {
    title: '',
    key: 'actions',
    width: 100,
    render: (row) =>
      h(
        'button',
        {
          class: 'pick-btn',
          onClick: () => pick(row),
        },
        t('manual_import.select')
      ),
  },
];

const previewColumns: DataTableColumns<AnyMapping> = [
  { title: () => t('manual_import.source'), key: 'source_path', ellipsis: { tooltip: true } },
  { title: () => t('manual_import.target'), key: 'target_path', ellipsis: { tooltip: true } },
];

const resultColumns: DataTableColumns<ImportApplyResult> = [
  { title: () => t('manual_import.target'), key: 'target_path', ellipsis: { tooltip: true } },
  {
    title: () => t('manual_import.result'),
    key: 'succeeded',
    width: 100,
    render: (row) =>
      row.succeeded
        ? t('manual_import.result_ok')
        : `${t('manual_import.result_failed')}${row.detail ? `: ${row.detail}` : ''}`,
  },
];
</script>

<template>
  <ab-modal v-model:show="show" :title="$t('manual_import.title')">
    <!-- Step 1: pick an unmanaged torrent, or point at a folder directly -->
    <div v-if="step === 'pick'" class="import-step">
      <div class="mode-tabs">
        <button
          type="button"
          class="mode-tab"
          :class="{ active: mode === 'torrent' }"
          @click="mode = 'torrent'"
        >
          {{ $t('manual_import.mode_torrent') }}
        </button>
        <button
          type="button"
          class="mode-tab"
          :class="{ active: mode === 'folder' }"
          @click="mode = 'folder'"
        >
          {{ $t('manual_import.mode_folder') }}
        </button>
        <button
          type="button"
          class="mode-tab"
          :class="{ active: mode === 'upload' }"
          @click="mode = 'upload'"
        >
          {{ $t('manual_import.mode_upload') }}
        </button>
      </div>

      <template v-if="mode === 'torrent'">
        <p class="import-hint">{{ $t('manual_import.pick_hint') }}</p>
        <NSpin :show="loading.candidates">
          <NDataTable
            :columns="candidateColumns"
            :data="candidates"
            :row-key="(row: ImportCandidate) => row.hash"
            :pagination="false"
            size="small"
          />
          <div v-if="!loading.candidates && candidates.length === 0" class="import-empty">
            {{ $t('manual_import.no_candidates') }}
          </div>
        </NSpin>
      </template>

      <template v-else-if="mode === 'folder'">
        <p class="import-hint">{{ $t('manual_import.folder_hint') }}</p>
        <div class="form-group">
          <label class="form-label">{{ $t('manual_import.folder_path') }}</label>
          <input
            v-model="folderPath"
            type="text"
            class="form-input"
            :placeholder="$t('manual_import.folder_path_placeholder')"
          />
        </div>
      </template>

      <template v-else>
        <p class="import-hint">{{ $t('manual_import.upload_hint') }}</p>
        <div class="form-group">
          <label class="form-label">{{ $t('manual_import.upload_file') }}</label>
          <input type="file" accept=".torrent" @change="onFileSelected" />
        </div>
      </template>
    </div>

    <!-- Step 2: show name + season -->
    <div v-else-if="step === 'form'" class="import-step">
      <div class="form-group">
        <label class="form-label">{{ $t('manual_import.show_name') }}</label>
        <input v-model="officialTitle" type="text" class="form-input" />
      </div>
      <div class="form-group">
        <label class="form-label">{{ $t('manual_import.season') }}</label>
        <NInputNumber v-model:value="season" :min="0" :max="99" />
      </div>
    </div>

    <!-- Step 3: preview -->
    <div v-else-if="step === 'preview' && preview" class="import-step">
      <p class="import-hint">
        {{ $t('manual_import.destination') }}: <code>{{ preview.target_folder }}</code>
      </p>
      <NDataTable
        :columns="previewColumns"
        :data="preview.mappings"
        :row-key="(row: AnyMapping) => row.source_path"
        :pagination="false"
        size="small"
      />
      <div v-if="preview.unparsed.length > 0" class="import-warning">
        {{ $t('manual_import.unparsed_warning', [preview.unparsed.length]) }}
        <ul>
          <li v-for="name in preview.unparsed" :key="name">{{ name }}</li>
        </ul>
      </div>
    </div>

    <!-- Step 4: result -->
    <div v-else-if="step === 'result'" class="import-step">
      <NDataTable
        :columns="resultColumns"
        :data="results"
        :row-key="(row: ImportApplyResult) => row.target_path"
        :pagination="false"
        size="small"
      />
    </div>

    <template #footer>
      <template v-if="step === 'pick'">
        <ab-button variant="secondary" size="sm" @click="close">
          {{ $t('setup.nav.cancel') }}
        </ab-button>
        <ab-button
          v-if="mode === 'folder'"
          variant="primary"
          size="sm"
          :disabled="!folderPath.trim()"
          @click="pickFolder"
        >
          {{ $t('setup.nav.next') }}
        </ab-button>
        <ab-button
          v-else-if="mode === 'upload'"
          variant="primary"
          size="sm"
          :loading="loading.upload"
          :disabled="!uploadFile"
          @click="pickUpload"
        >
          {{ $t('setup.nav.next') }}
        </ab-button>
      </template>
      <template v-else-if="step === 'form'">
        <ab-button variant="secondary" size="sm" @click="goBack">
          {{ $t('setup.nav.previous') }}
        </ab-button>
        <ab-button
          variant="primary"
          size="sm"
          :loading="loading.preview"
          @click="runPreview"
        >
          {{ $t('manual_import.preview_button') }}
        </ab-button>
      </template>
      <template v-else-if="step === 'preview'">
        <ab-button variant="secondary" size="sm" @click="goBack">
          {{ $t('setup.nav.previous') }}
        </ab-button>
        <ab-button
          variant="primary"
          size="sm"
          :loading="loading.apply"
          :disabled="!preview || preview.mappings.length === 0"
          @click="confirmApply"
        >
          {{ $t('manual_import.apply_button') }}
        </ab-button>
      </template>
      <template v-else-if="step === 'result'">
        <ab-button variant="primary" size="sm" @click="close">
          {{ $t('manual_import.done') }}
        </ab-button>
      </template>
    </template>
  </ab-modal>
</template>

<style lang="scss" scoped>
.import-step {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 200px;
}

.import-hint {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin: 0;
}

.mode-tabs {
  display: flex;
  gap: 4px;
  padding: 3px;
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  width: fit-content;
}

.mode-tab {
  height: 28px;
  padding: 0 12px;
  border-radius: calc(var(--radius-sm) - 2px);
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;

  &.active {
    background: var(--color-primary);
    color: var(--color-white);
  }
}

.import-empty {
  padding: 24px 0;
  text-align: center;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.import-warning {
  font-size: 12px;
  color: var(--color-warning-text);
  background: var(--color-warning-bg);
  border: 1px solid var(--color-warning-border);
  border-radius: var(--radius-sm);
  padding: 8px 12px;

  ul {
    margin: 4px 0 0;
    padding-left: 18px;
  }
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-label {
  font-size: 13px;
  font-weight: 500;
}

.form-input {
  height: 36px;
  padding: 0 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 14px;
}

:deep(.pick-btn) {
  height: 28px;
  padding: 0 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 12px;
  cursor: pointer;

  &:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }
}
</style>
