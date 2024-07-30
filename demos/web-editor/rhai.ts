import { LRLanguage, LanguageSupport } from "@codemirror/language";
import { styleTags, tags } from "@lezer/highlight";
import parser from "./rhai.grammar";

let parserWithMetadata = parser.configure({
  props: [
    styleTags({
      DefinitionKeyword: tags.definitionKeyword,
      ControlKeyword: tags.controlKeyword,
      "Call/identifier": tags.function(tags.name),
      Identifier: tags.name,
      Number: tags.number,
      String: tags.string,
    }),
  ],
});

export const rhaiLanguage = LRLanguage.define({
  parser: parserWithMetadata,
});

export function rhai() {
  return new LanguageSupport(rhaiLanguage, []);
}
