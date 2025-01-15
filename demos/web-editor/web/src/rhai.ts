import { LRLanguage, LanguageSupport } from "@codemirror/language";
import { styleTags, tags } from "@lezer/highlight";
import parser from "./rhai.grammar";

let parserWithMetadata = parser.configure({
  props: [
    styleTags({
      DefinitionKeyword: tags.definitionKeyword,
      "Call/Identifier": tags.function(tags.name),
      ControlKeyword: tags.controlKeyword,
      Identifier: tags.name,
      Number: tags.number,
      String: tags.string,
      LineComment: tags.comment,
    }),
  ],
});

export const rhaiLanguage = LRLanguage.define({
  parser: parserWithMetadata,
});

export function rhai() {
  return new LanguageSupport(rhaiLanguage, []);
}
