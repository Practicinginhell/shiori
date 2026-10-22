// Display text. Inuktitut is deliberately empty: it must be written and recorded with a
// fluent speaker from the community, never machine-translated.

import type { Lang, SewageStatus, StateName, Status } from "./types.js";

type Table = Record<string, string>;

const STATE_WORDS: Record<"en" | "fr", Record<StateName, string>> = {
  en: { protected: "Protected", check: "Check", boil: "Boil first", service: "Needs service" },
  fr: { protected: "Protégée", check: "À vérifier", boil: "Faire bouillir", service: "Entretien requis" },
};

const MESSAGES: Record<"en" | "fr", Table> = {
  en: {
    protected: "Water is protected.",
    no_chlorine_protection: "Chlorine protection is gone. Boil water for 2 minutes before drinking.",
    water_cloudy: "The water looks cloudy. Boil before drinking and call the municipal office.",
    tank_nearly_empty: "The tank is almost empty and the last water can carry sediment. Boil before drinking and ask for a delivery.",
    sensor_fault: "The device needs checking. Until then, follow the municipality's advice.",
    internal_error: "The device needs checking. Until then, follow the municipality's advice.",
    protection_fading: "Chlorine protection is getting low. Ask for a delivery soon.",
    water_cloudier: "The water is a little cloudier than at delivery.",
    water_running_low: "Water may run out before the next truck. Save water and ask for a delivery.",
  },
  fr: {
    protected: "L'eau est protégée.",
    no_chlorine_protection: "La protection au chlore est épuisée. Faites bouillir l'eau 2 minutes avant de la boire.",
    water_cloudy: "L'eau est trouble. Faites-la bouillir avant de la boire et appelez le bureau municipal.",
    tank_nearly_empty: "Le réservoir est presque vide et la dernière eau peut contenir des sédiments. Faites bouillir et demandez une livraison.",
    sensor_fault: "L'appareil doit être vérifié. D'ici là, suivez les consignes de la municipalité.",
    internal_error: "L'appareil doit être vérifié. D'ici là, suivez les consignes de la municipalité.",
    protection_fading: "La protection au chlore diminue. Demandez une livraison bientôt.",
    water_cloudier: "L'eau est un peu plus trouble qu'à la livraison.",
    water_running_low: "L'eau pourrait manquer avant le prochain camion. Économisez l'eau et demandez une livraison.",
  },
};

const SEVERITY: Record<StateName, number> = { protected: 0, check: 1, service: 2, boil: 3 };
const REASON_STATE: Record<string, StateName> = {
  no_chlorine_protection: "boil", water_cloudy: "boil", tank_nearly_empty: "boil",
  sensor_fault: "service", internal_error: "service",
  protection_fading: "check", water_cloudier: "check", water_running_low: "check",
};

export const INUKTITUT_PENDING =
  "Inuktitut text and audio will be written and recorded with a fluent speaker from the community. " +
  "They are not machine-translated. Showing English until then.";

function textLang(lang: Lang): "en" | "fr" {
  return lang === "fr" ? "fr" : "en";
}

export function stateWord(state: StateName, lang: Lang): string {
  return STATE_WORDS[textLang(lang)][state];
}

// The message for the state being shown: the most severe reason at that state's level.
export function message(status: Status, lang: Lang): string {
  const table = MESSAGES[textLang(lang)];
  const match = status.reasons.find((code) => REASON_STATE[code] === status.state);
  if (match) return table[match];
  if (status.state === "protected") return table.protected;
  const worst = [...status.reasons].sort((a, b) => SEVERITY[REASON_STATE[b]] - SEVERITY[REASON_STATE[a]])[0];
  return worst ? table[worst] : table.sensor_fault;
}

// A second warning at a lower level (e.g. low water while the light is red for chlorine), so the
// most severe message never hides the next thing the family should do.
export function alsoMessage(status: Status, lang: Lang): string | null {
  // Only while the light itself is not green, and never above the light's own level: one noisy
  // reading must not sneak an alarm past the two-readings rule.
  if (status.state === "protected") return null;
  const table = MESSAGES[textLang(lang)];
  const main = message(status, lang);
  const other = status.reasons.find((code) => table[code] && table[code] !== main && REASON_STATE[code] !== "service"
    && SEVERITY[REASON_STATE[code]] <= SEVERITY[status.state]);
  if (!other) return null;
  return (textLang(lang) === "fr" ? "Aussi : " : "Also: ") + table[other];
}

export function details(status: Status, lang: Lang): string[] {
  const fr = textLang(lang) === "fr";
  const lines: string[] = [];
  if (status.hours_protected !== null) {
    const h = Math.round(status.hours_protected);
    lines.push(status.hours_protected <= 0
      ? (fr ? "Aucune protection au chlore" : "No chlorine protection now")
      : fr ? `Protégée encore environ ${h} h` : `Protected for about ${h} more hours`);
  }
  if (status.days_left !== null) {
    const d = status.days_left.toFixed(1);
    lines.push(fr ? `Eau pour environ ${d} jours` : `Water for about ${d} days`);
  }
  if (status.runout_risk !== null && status.runout_risk > 0) {
    const p = Math.round(status.runout_risk * 100);
    lines.push(fr ? `Risque de manquer d'eau avant le camion : ${p} %` : `Chance of running out before the truck: ${p}%`);
  }
  if (status.fc_mg_l !== null) {
    const c = status.fc_mg_l.toFixed(2);
    lines.push(fr ? `Chlore estimé : ${c} mg/L` : `Estimated chlorine: ${c} mg/L`);
  }
  return lines;
}

const SEWAGE_WORDS: Record<"en" | "fr", Record<SewageStatus["state"], string>> = {
  en: { ok: "OK", soon: "Pump-out soon", urgent: "Pump out now", service: "Needs service" },
  fr: { ok: "OK", soon: "Vidange bientôt", urgent: "Vidange urgente", service: "Entretien requis" },
};

const SEWAGE_MESSAGES: Record<"en" | "fr", Table> = {
  en: {
    ok: "No pump-out needed yet.",
    sewage_full: "The sewage tank is full and can back up into the house. Use as little water as you can and call for a pump-out.",
    sewage_freezing: "The sewage tank is close to freezing. Check the tank heater and call for a pump-out.",
    sewage_filling: "The sewage tank will be full soon. Ask for a pump-out.",
    sewage_sensor_fault: "The sewage level sensor needs checking.",
  },
  fr: {
    ok: "Pas besoin de vidange pour l'instant.",
    sewage_full: "Le réservoir d'eaux usées est plein et peut refouler dans la maison. Utilisez le moins d'eau possible et demandez une vidange.",
    sewage_freezing: "Le réservoir d'eaux usées risque de geler. Vérifiez le chauffage du réservoir et demandez une vidange.",
    sewage_filling: "Le réservoir d'eaux usées sera bientôt plein. Demandez une vidange.",
    sewage_sensor_fault: "Le capteur de niveau des eaux usées doit être vérifié.",
  },
};

export function sewageTitle(lang: Lang): string {
  return textLang(lang) === "fr" ? "Réservoir d'eaux usées (extérieur)" : "Sewage tank (outside)";
}

export function sewageWord(state: SewageStatus["state"], lang: Lang): string {
  return SEWAGE_WORDS[textLang(lang)][state];
}

export function sewageMessage(sewage: SewageStatus, lang: Lang): string {
  const table = SEWAGE_MESSAGES[textLang(lang)];
  return sewage.reasons.length ? table[sewage.reasons[0]] : table.ok;
}

export function sewageDetails(sewage: SewageStatus, lang: Lang): string {
  const fr = textLang(lang) === "fr";
  const parts: string[] = [];
  if (sewage.level_pct !== null) parts.push(fr ? `${Math.round(sewage.level_pct)} % plein` : `${Math.round(sewage.level_pct)}% full`);
  if (sewage.days_to_full !== null) {
    const d = sewage.days_to_full.toFixed(1);
    parts.push(fr ? `plein dans environ ${d} jours` : `full in about ${d} days`);
  }
  if (sewage.temp_c !== null) parts.push(`${Math.round(sewage.temp_c)} °C`);
  return parts.join(" · ");
}

export function speechLang(lang: Lang): string {
  return lang === "fr" ? "fr-CA" : "en-CA";
}
