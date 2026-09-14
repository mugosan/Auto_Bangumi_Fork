<script lang="ts" setup>
import type { RssParserLang, Tvdb } from '#/config';
import type { SettingItem } from '#/components';

const { t } = useMyI18n();
const { getSettingGroup } = useConfigStore();

const tvdb = getSettingGroup('tvdb');
const langs: RssParserLang = ['zh', 'en', 'jp'];

const items: SettingItem<Tvdb>[] = [
  {
    configKey: 'api_key',
    label: () => t('config.tvdb_set.api_key'),
    type: 'input',
    prop: {
      type: 'password',
      placeholder: 'thetvdb.com API key',
    },
  },
  {
    configKey: 'language',
    label: () => t('config.tvdb_set.language'),
    type: 'select',
    prop: {
      items: langs,
    },
  },
];
</script>

<template>
  <ab-fold-panel :title="$t('config.tvdb_set.title')">
    <div space-y-8>
      <p class="tvdb-hint">{{ $t('config.tvdb_set.hint') }}</p>
      <ab-setting
        v-model:data="tvdb.enable"
        config-key="enable"
        :label="() => t('config.tvdb_set.enable')"
        type="switch"
      />
      <template v-if="tvdb.enable">
        <ab-setting
          v-for="i in items"
          :key="i.configKey"
          v-bind="i"
          v-model:data="tvdb[i.configKey]"
        ></ab-setting>
      </template>
    </div>
  </ab-fold-panel>
</template>

<style lang="scss" scoped>
.tvdb-hint {
  font-size: 12px;
  color: var(--color-text-secondary);
  line-height: 1.5;
  margin: 0;
}
</style>
