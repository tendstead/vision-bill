<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import { t } from '$lib/i18n';
	import sourceGuide from '../../../../../paddleocr-service/README.md?url&no-inline';
	import bundleGuide from '../../../../../distribution/README.md?url&no-inline';
</script>

<svelte:head>
	<title>{$t('settings.dedicatedOcr.setupGuide')} · Vision Bill</title>
</svelte:head>

<article class="max-w-2xl space-y-5">
	<a class="text-sm underline" href={resolve('/settings')}>{$t('nav.settings')}</a>
	<h1 class="text-2xl font-semibold">{$t('settings.dedicatedOcr.setupGuide')}</h1>
	<section class="space-y-2">
		<h2 class="text-lg font-semibold">{$t('settings.dedicatedOcr.guide.setupTitle')}</h2>
		<p>{$t('settings.dedicatedOcr.guide.topology')}</p>
		<p>{$t('settings.dedicatedOcr.guide.firstReceipt')}</p>
	</section>
	<section class="space-y-2">
		<h2 class="text-lg font-semibold">{$t('settings.dedicatedOcr.guide.recoveryTitle')}</h2>
		<p>{$t('settings.dedicatedOcr.guide.retry')}</p>
		<p>{$t('settings.dedicatedOcr.guide.ownership')}</p>
		<p>
			<a href={asset('/vision-bill-operator-guides.zip')} download class="underline">
				{$t('settings.dedicatedOcr.guide.archive')} (.zip)
			</a>
		</p>
		<p>{$t('settings.dedicatedOcr.guide.extract')}</p>
		<ul class="list-disc pl-5">
			<li>
				<a href={sourceGuide} rel="external" download="vision-bill-ocr-source.md" class="underline">
					{$t('settings.dedicatedOcr.guide.source')} (.md)
				</a>
			</li>
			<li>
				<a href={bundleGuide} rel="external" download="vision-bill-ocr-bundle.md" class="underline">
					{$t('settings.dedicatedOcr.guide.bundle')} (.md)
				</a>
			</li>
		</ul>
	</section>
</article>
